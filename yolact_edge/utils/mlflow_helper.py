import json
import logging
import os

import mlflow
import torch

from yolact_edge.data import Config
from yolact_edge.utils.pytorch_export_contrib_ops import register as register_cus_op
from yolact_edge.utils.pytorch_export_contrib_ops import unregister as unregister_cus_op

logger = logging.getLogger("yolact.helper")


class MlFlowHelper(object):
    def __init__(self, expid=os.getenv('EXPID', 'yolact_edge')):
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        mlflow.set_experiment(expid)
        self._run = mlflow.start_run()
        print("====== start experiment with storage s3://mlflow/%s" % self._run.info.experiment_id)
        print("====== start run run_id: %s" % self._run.info.run_id)

    def log_params(self, config):
        def to_desc(val, prefix=None):
            params = {}
            for k, v in vars(val).items():
                if isinstance(v, Config):
                    for k1, v1 in json.loads(v.desc()).items():
                        params['%s.%s' % (k, k1)] = v1
                    continue
                params[k if prefix is None else '%s.%s' % (prefix, k)] = v
            return params

        for k, v in to_desc(config).items():
            mlflow.log_param(k, v)

    def log_metrics(self, key, value, step=0):
        mlflow.log_metric(key, value, step=step)

    def log_model(self, net, name):
        mlflow.pytorch.log_model(net, name)

    def export_onnx(self, net, name, input_img=None, input_names=['input'], input_shape=(1, 3, 500, 500)):
        net._export_extras = {"backbone": "full",
                              "interrupt": False,
                              "keep_statistics": False,
                              "moving_statistics": None}

        net.training = False
        if input_img is not None:
            from yolact_edge.utils.augmentations import FastBaseTransform
            input_img = torch.from_numpy(input_img).float()
            if torch.cuda.is_available():
                input_img = input_img.cuda()

            input_data = FastBaseTransform()(input_img.unsqueeze(0))
            input_shape = input_data.shape
        else:
            input_data = torch.randn(*list(input_shape))
        extras = {"backbone": "full", "interrupt": False, "keep_statistics": False, "moving_statistics": None}

        preds = net(input_data, extras=extras)["pred_outs"]
        name = '%s-%dx%d.onnx' % (name[:-5] if name.endswith('.onnx') else name, input_shape[3], input_shape[2])
        output_onnx = os.path.join('/tmp', name)
        try:
            register_cus_op()
            torch.onnx.export(
                net,
                input_data,
                output_onnx,
                opset_version=11,
                input_names=input_names,
                output_names=["pred_outs"],
            )
            mlflow.log_artifact(output_onnx, 'export')
        finally:
            unregister_cus_op()

        if input_img is not None:
            from eval import prep_display
            import cv2
            img_numpy = prep_display(preds, input_img, None, None, undo_transform=False)
            cv2.imwrite('/root/yolact_edge/save_out.jpg', img_numpy)

    def log_artifact(self, fp, path='model_out'):
        mlflow.log_artifact(fp, path)

    def register_model(self, artifact_path, name):
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=self._run.info.run_id, artifact_path=artifact_path)
        mlflow.register_model(model_uri, name)
