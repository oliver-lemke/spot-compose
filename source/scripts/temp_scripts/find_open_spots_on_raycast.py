import numpy as np


def steps_to_change_from_right(rays: np.ndarray) -> np.ndarray:
    """
    Calculate - at each position -  the number of steps it takes to get to the next
    value (to the left) that is different to the value at the current position
    :param rays: boolean array
    :return: see above
    """
    change = np.abs(np.diff(rays, axis=-1, prepend=0)).astype(bool)
    counting_vector = np.arange(rays.shape[-1])
    counting_vector = np.tile(counting_vector, (*rays.shape[:-1], 1))
    index_vector = np.zeros(rays.shape)
    index_vector[change] = counting_vector[change]
    extended_array = np.maximum.accumulate(index_vector, axis=-1)
    return counting_vector - extended_array


def steps_to_change(rays):
    from_right = steps_to_change_from_right(rays)
    from_left = steps_to_change_from_right(np.flip(rays, axis=-1))
    from_left = np.flip(from_left, axis=-1)
    min_from_both = np.minimum(from_right, from_left)
    return min_from_both


def main():
    # Example usage:
    input_array = np.array(
        [
            [True, False, False, False, False, True, True, True, False],
            [False, False, False, False, False, True, False, True, False],
        ]
    )
    result = steps_to_change(input_array)
    print(result)  # Output: [1 4 3 2 1 3 2 1 1]


if __name__ == "__main__":
    main()
