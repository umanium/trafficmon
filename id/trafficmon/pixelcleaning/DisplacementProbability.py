from id.trafficmon.pixelcleaning.PixelCleaningAbstract import PixelCleaningAbstract

__author__ = 'Luqman'


class DisplacementProbability(PixelCleaningAbstract):
    reference_image = None

    def __init__(self, image):
        PixelCleaningAbstract.__init__(self, "DisplacementProbability")
        self.reference_image = image

    def apply(self, image):
        # geser reference image (melambangkan kiri-kanan)
        self.reference_image = image
