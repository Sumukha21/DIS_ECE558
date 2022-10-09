import numpy as np
import datetime


class ShortestDistance:
    """
    Class for computing the shortest distance between two pixels in a given image with constraints on the type of adjacency
    to be used
    """
    def __init__(self, img, p, q, V, adjacency):
        """
        param img:
            Single channel 2D array
            type: ndarray
        param p:
            The location of source pixel
            type: List of 2 values [row, column] where 0 <= row <= img.shape[0] - 1 and 0 <= column <= img.shape[1] - 1
        param q:
            The location of destination pixel
            type: List of 2 values [row, column] where 0 <= row <= img.shape[0] - 1 and 0 <= column <= img.shape[1] - 1
        param V:
            The set of intensity values to be considered when computing adjacency with neighbors
            type: List of integer values. V belong to set of unique values present in the image array
        param adjacency:
            The type of adjacency to be considered when computing the shortest path
            type: string belonging to ["m", "4", "8"] where "m", "4" and "8" corresponds to m, 4 and 8 adjacency
        """
        self.img = img
        self.p = tuple(p)
        self.q = tuple(q)
        self.adjacency = adjacency
        self.V = V
        try:
            self.check_input_arguments()
        except AttributeError as e:
            print(e)
            exit()
        self.distances = np.full_like(img, 10**6)
        self.paths = dict()
        self.shortest_distance = self.find_distance_reverse(self.q, [])

    def check_input_arguments(self):
        if not len(set(self.V).intersection(set(np.unique(self.img)))):
            raise AttributeError("The set of pixel intensity values in V are invalid. No path exists."
                                 " Given set of values: ", self.V)
        if self.p[0] >= self.img.shape[0] or self.p[0] < 0 or self.p[1] >= self.img.shape[1] or self.p[1] < 0:
            raise AttributeError("The pixel location for point p is invalid. Please rectify and try again!")
        if self.q[0] >= self.img.shape[0] or self.q[0] < 0 or self.q[1] >= self.img.shape[1] or self.q[1] < 0:
            raise AttributeError("The pixel location for point q is invalid. Please rectify and try again!")
        if self.img[self.p] not in self.V:
            raise AttributeError("The value at the source pixel location (p) is not part of the set of V."
                                 " Naturally no path would exist!")
        if self.img[self.q] not in self.V:
            raise AttributeError("The value at the destination pixel location (q) is not part of the set of V. "
                                 " Naturally no path would exist!")
        if self.adjacency.lower() not in ["8", "4", "m"]:
            raise AttributeError("Provided option '%s' for adjacency is not a valid option."
                                 " Please check the doc string for valid options." % self.adjacency)
        if self.p == self.q:
            raise AttributeError("The source and destination point are the same! So distance would be 0."
                                 "No cyclic paths are currently supported by my code")

    def find_distance_reverse(self, q, paths, parent=None):
        """
        param q:
            Given this pixel location, we find the shortest path to the source pixel p
            type: List of 2 values [row, column] where 0 <= row <= img.shape[0] - 1 and 0 <= column <= img.shape[1] - 1
        param paths:
            List of pixel locations, which have been traversed already
            type: List of pixel locations
        param parent:
            The pixel location from which the recursion call was made
        return: the path containing the pixels traversed to connect to source to the destination
        """
        if q == self.p:
            self.paths["%d_%d" % (q[0], q[1])] = [self.p]
            self.distances[self.p[0], self.p[1]] = 0
            return [self.p]
        adjacent_neighbors = self.find_adjacent_neighbors(q)
        if not len(adjacent_neighbors) or adjacent_neighbors == [parent]:
            return None
        adjacent_neighbors = list(set(adjacent_neighbors).difference(set(paths + [parent])))
        if not len(adjacent_neighbors):
            return []
        final_path = paths
        no_path = True
        for i, adjacent_neighbor in enumerate(adjacent_neighbors):
            if self.distances[adjacent_neighbor[0], adjacent_neighbor[1]] == -1:
                continue
            elif self.distances[adjacent_neighbor[0], adjacent_neighbor[1]] != 10**6:
                path_length = self.distances[adjacent_neighbor[0], adjacent_neighbor[1]] + 1
                paths_i = self.paths["%d_%d" % (adjacent_neighbor[0], adjacent_neighbor[1])] + [q]
            else:
                paths_extension = self.find_distance_reverse(adjacent_neighbor, list(set(paths + adjacent_neighbors)), q)
                if paths_extension is None:
                    self.distances[adjacent_neighbor[0], adjacent_neighbor[1]] = -1
                    continue
                elif len(paths_extension) == 0:
                    continue
                paths_i = paths_extension + [q]
                path_length = len(paths_i) - 1
            if path_length < self.distances[q[0], q[1]]:
                self.distances[q[0], q[1]] = path_length
                self.paths["%d_%d" % (q[0], q[1])] = paths_i
                final_path = paths_i
                no_path = False
        if no_path:
            return None
        return final_path

    def find_adjacent_neighbors(self, pixel_loc):
        """
        Given the pixel location, this function returns the valid neighbors to be considered for finding the path
        :param pixel_loc:
            The pixel location for which the neighbors need to be selected
        :return: List of neighbors selected based on the adjacency
        """
        neighbors_4 = [(pixel_loc[0] - 1, pixel_loc[1]), (pixel_loc[0] + 1, pixel_loc[1]),
                       (pixel_loc[0], pixel_loc[1] - 1), (pixel_loc[0], pixel_loc[1] + 1)]
        neighbors_d = [(pixel_loc[0] - 1, pixel_loc[1] - 1), (pixel_loc[0] + 1, pixel_loc[1] + 1),
                       (pixel_loc[0] + 1, pixel_loc[1] - 1), (pixel_loc[0] - 1, pixel_loc[1] + 1)]
        neighbors_4 = [i for i in neighbors_4 if not (i[0] >= self.img.shape[0] or i[0] < 0 or i[1] >= self.img.shape[1] or i[1] < 0)]
        neighbors_d = [i for i in neighbors_d if not (i[0] >= self.img.shape[0] or i[0] < 0 or i[1] >= self.img.shape[1] or i[1] < 0)]
        if self.adjacency == "4":
            neighbors = neighbors_4
        else:
            neighbors = neighbors_4 + neighbors_d
        adjacent_neighbors = []
        if self.adjacency == "4" or self.adjacency == "8":
            for neighbor in neighbors:
                if self.img[neighbor[0], neighbor[1]] in self.V:
                    adjacent_neighbors.append(neighbor)
        else:
            for neighbor in neighbors_4:
                if self.img[neighbor[0], neighbor[1]] in self.V:
                    adjacent_neighbors.append(neighbor)

            for neighbor in neighbors_d:
                if self.img[neighbor[0], neighbor[1]] in self.V:
                    neighbors_4_of_neighbor = [(neighbor[0] - 1, neighbor[1]), (neighbor[0] + 1, neighbor[1]),
                                               (neighbor[0], neighbor[1] - 1), (neighbor[0], neighbor[1] + 1)]
                    neighbors_4_intersection = set(adjacent_neighbors).intersection(set(neighbors_4_of_neighbor))
                    if not len(neighbors_4_intersection):
                        adjacent_neighbors.append(neighbor)
        return adjacent_neighbors


if __name__ == "__main__":
    img_array = np.array([[3, 1, 2, 1], [2, 2, 0, 2], [1, 2, 1, 1], [1, 0, 1, 2]])
    # img_array = np.ones((9, 9))
    p = [1, 3]
    q = [1, 3]
    V = [1, 2]
    adjacency = "4"
    start = datetime.datetime.now()
    distance_finder = ShortestDistance(img_array, p, q, V, adjacency)
    path = distance_finder.shortest_distance
    if path is not None:
        distance = len(distance_finder.shortest_distance) - 1
    else:
        print("No path exists")
        exit()
    print("The distance and the corresponding path via locations: ", distance, path)
    print("The path according to pixel values:", [img_array[i] for i in path])
    print("Time taken to run the code: %0.2f microseconds" % ((datetime.datetime.now()) - start).microseconds)
