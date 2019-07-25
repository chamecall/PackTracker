from collections import deque


def are_rectangles_intersect(first_rectangle, second_rectangle, shift=0):
    if first_rectangle[0][0] > second_rectangle[1][0] + shift or second_rectangle[0][0] > \
            first_rectangle[1][0] + shift:
        return False

    if first_rectangle[0][1] > second_rectangle[1][1] + shift or second_rectangle[0][1] > \
            first_rectangle[1][1] + shift:
        return False
    return True


def are_rectangle_inside_another(prob_in_rectangle, prob_out_rectangle):
    in_xs = [point[0] for point in prob_in_rectangle]
    in_ys = [point[1] for point in prob_in_rectangle]

    out_min_x, out_max_x = min(prob_out_rectangle[0][0], prob_out_rectangle[1][0]), max(prob_out_rectangle[0][0], prob_out_rectangle[1][0])
    out_min_y, out_max_y = min(prob_out_rectangle[0][1], prob_out_rectangle[1][1]), max(prob_out_rectangle[0][1],
                                                                                        prob_out_rectangle[1][1])

    in_xs_are_inside_out_xs = all([out_min_x <= in_x <= out_max_x for in_x in in_xs])
    in_ys_are_inside_out_ys = all([out_min_y <= in_y <= out_max_y for in_y in in_ys])
    return in_xs_are_inside_out_xs and in_ys_are_inside_out_ys

def rectangles_intersection(rect_a, rect_b):
    min_left_x, max_left_x = min(rect_a[0][0], rect_b[0][0]), max(rect_a[0][0], rect_b[0][0])
    min_right_x, max_right_x = min(rect_a[1][0], rect_b[1][0]), max(rect_a[1][0], rect_b[1][0])

    min_left_y, max_left_y = min(rect_a[0][1], rect_b[0][1]), max(rect_a[0][1], rect_b[0][1])
    min_right_y, max_right_y = min(rect_a[1][1], rect_b[1][1]), max(rect_a[1][1], rect_b[1][1])

    return (max_left_x, max_left_y), (min_right_x, min_right_y)

def combine_nearby_rects(rectangles, shift=0):
    if not rectangles:
        return None

    rectangles = deque(rectangles)
    num_permutations = 0
    epochs = 0
    base_rectangle = rectangles.popleft() if rectangles else None
    while rectangles:
        rectangle = rectangles.popleft()
        if are_rectangles_intersect(base_rectangle, rectangle, shift=shift):
            base_rectangle = ((min(base_rectangle[0][0], rectangle[0][0]), min(base_rectangle[0][1], rectangle[0][1])),
                              (max(base_rectangle[1][0], rectangle[1][0]), max(base_rectangle[1][1], rectangle[1][1])))
            num_permutations = 0
        else:
            rectangles.append(rectangle)
            num_permutations += 1

        num_rectangles = len(rectangles)
        if num_permutations >= num_rectangles:
            rectangles.append(base_rectangle)
            base_rectangle = rectangles.popleft()
            num_permutations = 0
            epochs += 1

        if epochs >= num_rectangles:
            rectangles.append(base_rectangle)
            break
    else:
        rectangles.append(base_rectangle)
    return list(rectangles)

def restrict_rectangle_areas_by_another_ones(areas, bounding_areas):
    if not areas:
        return []

    if not bounding_areas:
        return areas

    final_areas = []
    for area in areas:
        for bounding_area in bounding_areas:
            if are_rectangles_intersect(area, bounding_area):
                final_areas.append(rectangles_intersection(area, bounding_area))
    return final_areas


def is_num_in_range(num, num_range: tuple):
    return num_range[0] <= num <= num_range[1]