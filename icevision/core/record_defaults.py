__all__ = [
    "ObjectDetectionRecord",
    "InstanceSegmentationRecord",
    "SemanticSegmentationRecord",
    "KeypointsRecord",
]

from icevision.core.record import *
from icevision.core.record_components import *
from icevision.core import tasks


def ObjectDetectionRecord():
    return BaseRecord(
        (
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
        )
    )


def InstanceSegmentationRecord():
    return BaseRecord(
        (
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            InstanceMasksRecordComponent(),
        )
    )


def SemanticSegmentationRecord(gray=False):
    return BaseRecord(
        (
            FilepathRecordComponent(gray=gray),
            ClassMapRecordComponent(task=tasks.segmentation),
            SemanticMaskRecordComponent(),
        )
    )


def KeypointsRecord():
    return BaseRecord(
        (
            FilepathRecordComponent(),
            InstancesLabelsRecordComponent(),
            BBoxesRecordComponent(),
            KeyPointsRecordComponent(),
        )
    )
