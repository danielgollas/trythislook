# all of this is just a refactoring on this great work by Matthew Earl
# https://matthewearl.github.io/2015/07/28/switching-eds-with-python/

import logging
import os

import cv2
import cv
import dlib
import numpy

import exceptions

logger = logging.getLogger(__name__)

RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))

class NeedMoreHeadsException(Exception):
    pass


class TooManyFaces(Exception):
    pass


class TooFewFaces(Exception):
    pass


class NoDetector(Exception):
    pass


class FaceImage(object):

    MASK_SHAPE_T = 0
    MASK_SHAPE_OVAL = 1

    def __init__(self, path=None, scale_factor=1, detector=None, predictor=None, max_size = 1024):
        self.scale_factor = scale_factor
        self.path = path
        self.img = None
        self.steps = {
            "original" : None,
            "annotated" : None,
            "transformed" : None,
            "mask": None,
            "transformed_mask": None,
            "composite": None
        }
        self.transform_matrix = None
        self.transformed_img = None
        self.annotated_img = None
        self.landmarks = None
        self.detector = detector
        self.predictor = predictor
        self.max_size = max_size

        if self.path:
            self.load()

    def show_transformed_img(self):
        cv2.imshow('image', self.transformed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_img(self):
        cv2.imshow('image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_img(self, step="original", dir=".", name=None, img_format="png", frame_number=0):
        if not name:
            name = os.path.splitext(os.path.basename(self.path))[0]

        name = name+"-" + step

        if step not in self.steps:
            raise ValueError("No step named %s"%(step))
        name = "%s_%04d.%s"%(name, frame_number, img_format)
        final_path = os.path.join(dir,name)
        cv2.imwrite(final_path, self.steps[step])

    def load(self):
        self.path = assert_file_exists(self.path)
        logger.info("Loading face: %s" % (self.path))
        self.img = cv2.imread(self.path)
        w, h = self.img.shape[1], self.img.shape[0]
        if w > self.max_size or h > self.max_size:
            #image is too big, resize to max size
            ratio = float(w) / float(h)
            if ratio > 1.0:
                #image wider than taller, make width = max_size
                new_w = int(self.max_size)
                new_h = int(self.max_size*ratio)


            else:
                #image taller than wider
                new_w = int(self.max_size * ratio)
                new_h = int(self.max_size)

            newShape = (new_w, new_h)
            self.steps["original"] = cv2.resize(self.img, newShape)
        else:
            self.steps["original"] = self.img.copy()

        #if self.scale_factor != 1:

        self.img = cv2.resize(self.img,
                                  (self.img.shape[1] * self.scale_factor, self.img.shape[0] * self.scale_factor))

    def find_landmarks(self):
        if not self.detector:
            raise NoDetector

        rects = self.detector(self.steps["original"], 1)
        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) == 0:
            raise TooFewFaces

        self.landmarks = numpy.matrix([[p.x, p.y] for p in self.predictor(self.steps["original"], rects[0]).parts()])
        self.annotate_landmarks()

    def annotate_landmarks(self):
        self.steps["annotated"] = self.steps["original"].copy()
        for idx, point in enumerate(self.landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(self.steps["annotated"], str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(self.steps["annotated"], pos, 3, color=(0, 255, 255))


    def get_points(self, points):
        return self.landmarks[points]

    def afine_transform(self, transform_matrix, shape):
        self.transform_matrix = transform_matrix
        self.steps["transformed"] = numpy.zeros(shape, dtype=self.steps["original"].dtype)
        cv2.warpAffine(self.steps["original"],
                       transform_matrix[:2],
                       (shape[1], shape[0]),
                       dst=self.steps["transformed"],
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP
                       )

    def draw_convex_hull(self, im, points, color):
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)
        return im

    def create_mask(self, shape, type=MASK_SHAPE_T, overlay_points=None,
                    feather_amount=11, color=1.0, scale=1.0, transform_matrix = None):

        mask = numpy.zeros(self.steps["original"].shape[:2], dtype=numpy.float64)

        for group in overlay_points:
            self.draw_convex_hull(mask,
                             self.landmarks[group],
                             color=color)

        mask = numpy.array([mask, mask, mask]).transpose((1, 2, 0))

        mask = (cv2.GaussianBlur(mask, (feather_amount, feather_amount), 0) > 0) * 1.0
        mask = cv2.GaussianBlur(mask, (feather_amount, feather_amount), 0)

        self.steps["mask"] = mask
        # self.steps["mask_rgb"] = cv2.cvtColor(mask, cv.CV_GRAY2RGB)

        if transform_matrix != None:
            transformed_mask = numpy.zeros(shape, dtype=mask.dtype)
            cv2.warpAffine(mask,
                           transform_matrix[:2],
                           (shape[1], shape[0]),
                           dst=transformed_mask,
                           borderMode=cv2.BORDER_TRANSPARENT,
                           flags=cv2.WARP_INVERSE_MAP
                           )

            self.steps["mask"] = transformed_mask

    def compose(self, other, mask):
        composite = self.steps["original"] * (1.0 - mask) + \
                    other * mask

        self.steps["composite"] = composite

    def color_correct(self, target, target_landmarks, blur=0.7 ):

        source = self.steps["transformed"]

        blur_amount = blur * numpy.linalg.norm(
                numpy.mean(target_landmarks[LEFT_EYE_POINTS], axis=0) -
                numpy.mean(target_landmarks[RIGHT_EYE_POINTS], axis=0))

        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(target, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(source, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

        self.steps["color_corrected"]=(source.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                im2_blur.astype(numpy.float64))

        # self.steps["color_corrected"] = (self.steps["transform"].astype(numpy.float64) * im1_blur.astype(numpy.float64) /
        #         im2_blur.astype(numpy.float64))

def assert_file_exists(file_path):
    file_path = os.path.abspath(file_path)
    if not os.path.exists(file_path):
        raise exceptions.IOError("%s does not exist!" % (file_path))
    return file_path


class Faceoff(object):

    FACE_POINTS = list(range(17, 68))
    MOUTH_POINTS = list(range(48, 61))
    RIGHT_BROW_POINTS = list(range(17, 22))
    LEFT_BROW_POINTS = list(range(22, 27))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    NOSE_POINTS = list(range(27, 35))
    JAW_POINTS = list(range(0, 17))
    CHIN_POINTS = list(range(5,11)+range(48,54))



    # Points used to line up the images.
    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                    RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS + JAW_POINTS)
    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                    RIGHT_BROW_POINTS + JAW_POINTS)

    # Points from the second image to overlay on the first. The convex hull of each
    OVERLAY_POINTS = [
        LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
        NOSE_POINTS + MOUTH_POINTS,
        CHIN_POINTS
    ]

    # element will be overlaid.

    def __init__(self, shape_predictor_path="shape_predictor_68_face_landmarks.dat", max_size=1024):
        self.shape_predictor_path = shape_predictor_path
        self.scale_factor = 1
        self.feather_amount = 25
        self.color_correct_blur = 0.6

        self.max_size = max_size

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        self.load_predictor(self.shape_predictor_path)

        self.face_image = None
        self.head_images = []

        # todo: Make this configurable.

        self.align_points = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS +
                             self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)
        self.overlay_points = [
            self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_BROW_POINTS + self.RIGHT_BROW_POINTS,
            self.NOSE_POINTS + self.MOUTH_POINTS,
        ]

    def faceoff(self, face_img_path, head_img_paths=[]):
        if len(head_img_paths) == 0:
            raise NeedMoreHeadsException



        self.load_images(face_img_path, head_img_paths)

        out_dir = os.path.dirname(self.face_image.path)
        out_dir = os.path.join(out_dir, "results")

        face_img_prefix = os.path.splitext(os.path.basename(self.face_image.path))[0]
        self.face_image.save_img("original", dir=out_dir, frame_number=0)

        self.face_image.find_landmarks()
        tmp_name = face_img_prefix+"_landmarks"

        self.face_image.save_img("annotated", dir = out_dir, frame_number=1)
        try:
            os.mkdir(out_dir)
        except Exception as e:
            pass

        for head_image in self.head_images:
            head_img_prefix = os.path.splitext(os.path.basename(head_image.path))[0]
            tmp_name = os.path.join(out_dir, "%s-%s"%(face_img_prefix,head_img_prefix))

            #FIND LANDMARKS
            head_image.find_landmarks()
            head_image.save_img("annotated", dir = out_dir, frame_number=2)

            #FIND ALIGNMENT MATRIX
            align_matrix = self.get_alignment_matrix(head_image, self.face_image)
            self.face_image.transform_matrix = align_matrix
            #align_matrix = self.get_alignment_matrix(self.face_image, head_image)

            # CREATE A FACE MASK
            self.face_image.create_mask(head_image.steps["original"].shape, type=FaceImage.MASK_SHAPE_T, overlay_points=self.OVERLAY_POINTS,
                                        feather_amount=self.feather_amount, transform_matrix=align_matrix)
            self.face_image.save_img(step="mask", name=tmp_name, dir=out_dir, frame_number=3)

            #CREATE A HEAD MASK
            head_image.transform_matrix = align_matrix
            head_image.create_mask(head_image.img.shape, type=FaceImage.MASK_SHAPE_T, overlay_points=self.OVERLAY_POINTS,
                                        feather_amount=self.feather_amount, transform_matrix=None)

            #TRANSFORM ORIGINAL IMAGE
            self.face_image.afine_transform(align_matrix, head_image.steps["original"].shape)
            self.face_image.save_img("transformed", name=tmp_name, dir= out_dir, frame_number=4)

            #COLOR CORRECT FACE
            self.face_image.color_correct(head_image.steps["original"], head_image.landmarks, blur=self.color_correct_blur)
            self.face_image.save_img("color_corrected", name=tmp_name, dir=out_dir, frame_number=5)

            #COMBINE FACE AND HEAD MASKS
            combined_mask = numpy.max([head_image.steps["mask"],
                                       self.face_image.steps["mask"]],
                                      axis=0)

            head_image.compose(self.face_image.steps["color_corrected"], combined_mask)
            # head_image.compose(self.face_image.steps["transformed"], self.face_image.steps["transformed_mask"])
            head_image.save_img(step="composite", name=tmp_name, dir=out_dir, frame_number=6)



    def load_predictor(self, path):
        logger.info("Loading predictor :%s" % path)
        self.predictor = dlib.shape_predictor(path)

    def load_images(self, face_img_path, head_img_paths=[]):

        self.face_image = FaceImage(path=face_img_path, detector=self.detector, predictor=self.predictor, max_size = self.max_size)
        logger.info("Loading %i heads" % (len(head_img_paths)))
        for head_img_path in head_img_paths:
            tmp_face_img = FaceImage(head_img_path, detector=self.detector, predictor=self.predictor, max_size=self.max_size)
            self.head_images.append(tmp_face_img)

    def get_alignment_matrix(self, face_a, face_b):

        points_a = face_a.get_points(self.ALIGN_POINTS)
        points_b = face_b.get_points(self.ALIGN_POINTS)

        points_a = points_a.astype(numpy.float64)
        points_b = points_b.astype(numpy.float64)

        # find centroids for the points
        c1 = numpy.mean(points_a, axis=0)
        c2 = numpy.mean(points_b, axis=0)
        # ...  and center all of them around it.
        points_a -= c1
        points_b -= c2

        # normalize the points using the standard deviation. Normalization lets us skip the scaling component of the
        # problem since they will both be normalized.
        s1 = numpy.std(points_a)
        s2 = numpy.std(points_b)
        points_a /= s1
        points_b /= s2

        U, S, Vt = numpy.linalg.svd(points_a.T * points_b)
        # Straight from the source: The R we seek is in fact the transpose of the one given by U * Vt. This
        # is because the above formulation assumes the matrix goes on the right
        # (with row vectors) where as our solution requires the matrix to be on the
        # left (with column vectors).
        R = (U * Vt).T
        return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                           c2.T - (s2 / s1) * R * c1.T)),
                             numpy.matrix([0., 0., 1.])])



if __name__ == "__main__":

    fo = Faceoff()

    face = "../sarav.jpg"
    # heads = ["../meghan.png","../girl.jpg"]
    heads = ["../jeremy.jpg"]

    fo.faceoff(face, heads)
