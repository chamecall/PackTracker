from Utils import is_num_in_range, calculate_two_sizes_match_precision
from PartDetection import PartDetection


class PartDetector:
    def __init__(self, precision_threshold=0.8):
        self.precision_threshold = precision_threshold

    def detect(self, pack_tasks, object_shapes, distance_coeff_calculator):
        part_detections = []
        best_precision_detections = []

        desired_parts = [pack_task.part for pack_task in pack_tasks if not pack_task.is_detected()]

        for desired_part in desired_parts:
            # define acceptable aspects deviations by the threshold value
            acceptable_width_deviation = int(desired_part.width * (1 - self.precision_threshold))
            acceptable_width_range = (
                desired_part.width - acceptable_width_deviation, desired_part.width + acceptable_width_deviation)

            acceptable_height_deviation = int(desired_part.height * (1 - self.precision_threshold))
            acceptable_height_range = (
                desired_part.height - acceptable_height_deviation,
                desired_part.height + acceptable_height_deviation)

            founded_parts = []
            for object_shape in object_shapes:
                distance_coeff = distance_coeff_calculator(object_shape.center)
                object_shape_width = object_shape.width * distance_coeff
                object_shape_height = object_shape.height * distance_coeff

                if (is_num_in_range(object_shape_width, acceptable_height_range) and
                        is_num_in_range(object_shape_height, acceptable_width_range)):
                    object_shape_width, object_shape_height = object_shape_height, object_shape_width

                elif not (is_num_in_range(object_shape_width, acceptable_width_range) and
                          is_num_in_range(object_shape_height, acceptable_height_range)):
                    break

                precision = calculate_two_sizes_match_precision((object_shape_width, object_shape_height),
                                                                (desired_part.width, desired_part.height))
                founded_parts.append((object_shape, precision))
            for founded_part in founded_parts:
                part_detections.append(PartDetection(desired_part, *founded_part))
        used_parts = []
        used_object_shapes = []
        # sort in desc by precision
        part_detections.sort(reverse=True, key=lambda detection: detection.precision)

        for part_detection in part_detections:
            the_part_amount = part_detection.part.multiplicity
            the_part_used_amount = sum([1 for used_part in used_parts if used_part is part_detection.part])
            if not (the_part_used_amount == the_part_amount or
                    part_detection.object_shape in used_object_shapes):
                best_precision_detections.append(part_detection)
                used_parts.append(part_detection.part)
                used_object_shapes.append(part_detection.object_shape)
        return best_precision_detections

