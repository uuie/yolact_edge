import datetime
import io
import json
import torch
import mlflow
import os

from yolact_edge.data import Config
from yolact_edge.yolact import Yolact


class MlFlowHelper(object):
    def __init__(self, expid=os.getenv('EXPID', 'yolact_edge')):
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
        mlflow.set_experiment(expid)
        self._run = mlflow.start_run()

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

    def export_onnx(self, net, name, input_names=['input'], input_shape=(1, 3, 280, 480)):
        net._export_extras = {"backbone": "full",
                              "interrupt": False,
                              "keep_statistics": False,
                              "moving_statistics": None}
        net.detect.use_fast_nms = False
        name = '%s-%dx%d.onnx' % (name[:-5] if name.endswith('.onnx') else name, input_shape[2], input_shape[3])
        output_onnx = os.path.join('/tmp', name)
        inputs = torch.randn(*input_shape).cuda()
        torch.onnx.export(
            net,
            inputs,
            output_onnx,
            opset_version=11,
            input_names=input_names,
            output_names=["pred_outs"]
        )
        mlflow.log_artifact(output_onnx, 'export')

    def log_artifact(self, fp, path='model_out'):
        mlflow.log_artifact(fp, path)

    def register_model(self, artifact_path, name):
        model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=self._run.info.run_id, artifact_path=artifact_path)
        mlflow.register_model(model_uri, name)


if __name__ == '__main__':
    net = Yolact()
    net.eval()
    net.to('cuda')
