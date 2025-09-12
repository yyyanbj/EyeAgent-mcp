import unittest
from unittest.mock import patch, MagicMock
import base64
from io import BytesIO
from PIL import Image
import os
import tempfile

from eyetools.tools.classification import (
    classify_image,
    classify_image_resnet,
    classify_image_vit,
    classify_image_fundus,
    modality,
    cfp_quality,
    laterality,
    pre_analyze,
    vis_prob,
)

class TestClassification(unittest.TestCase):

    def setUp(self):
        # Create a temporary image file for testing
        self.temp_dir = tempfile.mkdtemp()
        img = Image.new('RGB', (100, 100), color='red')
        self.image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        img.save(self.image_path)

    def tearDown(self):
        # Clean up temporary file
        if os.path.exists(self.image_path):
            os.remove(self.image_path)
        os.rmdir(self.temp_dir)

    def test_classify_image(self):
        result = classify_image(self.image_path)
        self.assertIsInstance(result, str)

    def test_classify_image_resnet(self):
        result = classify_image_resnet(self.image_path)
        self.assertIsInstance(result, str)

    def test_classify_image_vit(self):
        result = classify_image_vit(self.image_path)
        self.assertIsInstance(result, str)

    def test_classify_image_fundus(self):
        result = classify_image_fundus(self.image_path)
        self.assertIsInstance(result, str)

    def test_modality(self):
        result = modality(self.image_path)
        self.assertIsInstance(result, str)

    def test_cfp_quality(self):
        result = cfp_quality(self.image_path)
        self.assertIsInstance(result, str)

    def test_laterality(self):
        result = laterality(self.image_path)
        self.assertIsInstance(result, str)

    def test_pre_analyze(self):
        result = pre_analyze(self.image_path)
        self.assertIsInstance(result, list)

    def test_vis_prob(self):
        probabilities = [0.1, 0.5, 0.4]
        classes = ['class1', 'class2', 'class3']
        vis_prob(probabilities, classes)
        # No exception expected


if __name__ == '__main__':
    unittest.main()
