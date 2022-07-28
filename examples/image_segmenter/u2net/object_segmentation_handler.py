import io
import torch
import torchvision
import torchvision.transforms as T
from ts.torch_handler.vision_handler import VisionHandler


class ObjectSegmentationHandler(VisionHandler):
    image_processing = T.Compose([
        T.Resize(320),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    def inference(self, data, *args, **kwargs):
        """The Inference Request is made through this function and the user
        needs to override the inference function to customize it.

        Args:
            data (torch tensor): The data is in the form of Torch Tensor
                                    whose shape should match that of the
                                    Model Input shape.

        Returns:
            (Torch Tensor): The predicted response from the model is returned
                            in this function.
        """
        with torch.no_grad():
            results = self.model(data)
        return results[0]
        
    def postprocess(self, data):
        """
        The post process function converts the prediction response into a
            Torchserve compatible format

        Args:
            data (Torch Tensor): The data parameter comes from the prediction output
            output_explain (None): Defaults to None.

        Returns:
            (list): Returns the response containing the predictions and explanations
                    (if the Endpoint is hit).It takes the form of a list of dictionary.
        """

        def normPRED(d):
            ma = torch.max(d)
            mi = torch.min(d)

            dn = (d-mi)/(ma-mi)

            return dn

        pred = normPRED(data[:,0,:,:])
        
        with io.BytesIO() as fp:
            torchvision.utils.save_image(pred, fp, format="PNG")
            postprocessed_data = [fp.getvalue()]

        return postprocessed_data
