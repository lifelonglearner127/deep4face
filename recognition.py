import cv2
import mxnet as mx
import numpy as np
import sklearn
import settings


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


class FaceEmbedding:

    def __init__(self):
        ctx = mx.cpu() if settings.GPU == -1 else mx.gpu(settings.GPU)

        # Read pre-trained arcface embedding model
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            settings.ARCFACE_MODEL_PREFIX, settings.ARCFACE_EPOCH
        )
        all_layers = sym.get_internals()
        sym = all_layers['fc1_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[
            ('data', (1, 3, settings.IMAGE_SIZE[0], settings.IMAGE_SIZE[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def extract_face_embeddings(self, image_paths):
        result = []

        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            aligned = np.transpose(image, (2, 0, 1))
            embedding = None
            for flipid in [0, 1]:
                if flipid == 1:
                    do_flip(aligned)

                input_blob = np.expand_dims(aligned, axis=0)
                data = mx.nd.array(input_blob)
                db = mx.io.DataBatch(data=(data, ))
                self.model.forward(db, is_train=False)
                _embedding = self.model.get_outputs()[0].asnumpy()
                # print(_embedding.shape)
                if embedding is None:
                    embedding = _embedding
                else:
                    embedding += _embedding
            embedding = sklearn.preprocessing.normalize(embedding).flatten()
            result.append(embedding)

        return result

    def extract_face_embedding(self, aligned):
        aligned = np.transpose(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB),
                               (2, 0, 1))
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data, ))
        self.model.forward(db, is_train=False)
        embedding = self.model.get_outputs()[0].asnumpy()
        embedding = sklearn.preprocessing.normalize(embedding).flatten()
        return embedding
