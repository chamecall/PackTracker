from collections import deque
import cv2


class Animator:
    """Class performing an animation."""

    def __init__(self, animation_queue_size):
        self.animation_queue_size = animation_queue_size
        self.animation_queue = deque(maxlen=self.animation_queue_size)

        self.animation_min_line_thickness = 2
        self.animation_max_line_thickness = 13

    def generate_color_range(self, animation_initial_color, animation_final_color):
        color_step = self.define_bgr_color_step(animation_initial_color,
                                                animation_final_color)
        colors = [animation_initial_color]
        size = self.animation_queue_size - 1
        for i in range(size // 2):
            colors.append(self.add_color_step_to_color(colors[-1], color_step))
        for i in range(size - (size // 2)):
            colors.append(self.sub_color_step_from_color(colors[-1], color_step))

        return colors

    def generate_thickness_range(self, animation_min_font_thickness, animation_max_font_thickness):
        thickness_step = (animation_max_font_thickness - animation_min_font_thickness + 1) / self.animation_queue_size
        thicknesses = [animation_min_font_thickness]
        thickness = animation_min_font_thickness
        size = self.animation_queue_size - 1
        for i in range(size // 2):
            thickness += thickness_step
            thicknesses.append(int(thickness))
        for i in range(size - (size // 2)):
            thickness -= thickness_step
            thicknesses.append(int(thickness))
        return thicknesses

    def generate_animation(self, animation_initial_color, animation_final_color):
        thicknesses = self.generate_thickness_range(self.animation_min_line_thickness,
                                                    self.animation_max_line_thickness)
        colors = self.generate_color_range(animation_initial_color, animation_final_color)
        for i in range(self.animation_queue_size):
            self.animation_queue.append((thicknesses[i], colors[i]))

    def play_animation(self, frame, point_a, point_b):

        if self.animation_queue:
            thickness, color = self.animation_queue.pop()
            if point_a and point_b:
                cv2.line(frame, point_a, point_b, color, thickness)

    def define_bgr_color_step(self, initial_color: tuple, final_color: tuple):
        iters = int(self.animation_queue_size / 2)
        b_step = int((final_color[0] - initial_color[0]) / iters)
        g_step = int((final_color[1] - initial_color[1]) / iters)
        r_step = int((final_color[2] - initial_color[2]) / iters)
        return b_step, g_step, r_step

    @staticmethod
    def add_color_step_to_color(color: tuple, color_step: tuple):
        return [sum(color_pair) for color_pair in zip(color, color_step)]

    @staticmethod
    def sub_color_step_from_color(color: tuple, color_step: tuple):
        return [sum([color_pair[0], -color_pair[1]]) for color_pair in zip(color, color_step)]

    def is_animation_not_finished(self):
        return self.animation_queue
