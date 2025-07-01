# Ryan Peach 3/1/2016
# References Used for this Implementation
# https://en.wikipedia.org/wiki/Hungarian_algorithm
# https://github.com/bmc/munkres/blob/master/munkres.py
# http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html
# ---
# No copying and pasting of online code was performed, though some code may turn out to be similar due to the standardized nature of this algorithm.
# This is a very different implementation due to the fact that it heavily uses numpy, vastly simplifies many poorly pythonized coding elements
# removes the "class" approach for a function based one. Etc.
# Improvements that need to be made is to require the matrix class, and to vectorize iterations through the matrix.

import numpy as np
from queue import PriorityQueue
from time import time
from scipy.optimize import linear_sum_assignment
import heapq

# Module Globals
NONE, ZERO, STAR, PRIME = -1, 0, 1, 2
hungmem = {}


def score(C, answ):
    return sum([C[n, answ[n]] for n in range(len(answ))])


def hungarian(C):
    """Calculates the hungarian of C, remembering all values to avoid recalculation."""
    global hungmem  # Use module memory
    # print(C)
    C = np.matrix(C)  # Use a matrix
    H = C.tostring()
    if H not in hungmem:  # Check if it's in memory
        S = _hungarian(C)  # If it's not, calculate the solution
        hungmem[H] = S  # Then add the solution to memory
    else:  # If it is
        S = hungmem[H]  # simply retrieve the solution from memory
    return S  # Return the answer


def murty(P0):
    """Non-optimized Murty's. Generator.
    Ref: Optimizing Murty's Ranked Assignment Method, Fig. 4
    by Matt L. Miller, Harold S. Stone, & Ingemar J. Cox"""
    try:
        INF = np.iinfo(P0.dtype).max  # A value used to remove (y,z,l) from a problem
    except:
        INF = np.finfo(P0.dtype).max

    def valid(X):
        """Checks if any rows or columns are completely ignored."""
        for row in np.array(X):  # Iterate over all rows
            found = False
            for y in row:  # Iterate over all values in row
                if y != INF:  # If you find a cell that is not ignored
                    found = True
                    break  # Go on to the next row
            if not found:
                return False  # If you never find one, this is an invalid matrix

        for col in np.array(X.T):  # Iterate over all columns
            found = False
            for x in col:  # Iterate over all values in column
                if x != INF:  # If you find a cell that is not ignored
                    found = True
                    break  # Go on to the next column
            if not found:
                return False  # If you never find one, this is an invalid matrix

        return True

    S0 = hungarian(P0)  # Find the best solution S0 to problem P0 (1)
    C0 = score(P0, S0)  # Find the cost of S0 given problem P0
    Q = (
        PriorityQueue()
    )  # Initialize a priority queue (C, P, S) sorted top as lowest cost (2)
    Q.put(
        (C0, time(), P0.copy(), S0)
    )  # Add (C0, P0, S0) to the queue. Use time to sort equal costs
    # P0 will be modified, but the original is needed for score references.
    solutions, found = (
        set(),
        set(),
    )  # Create a searchable set of solutions. Found used to avoid duplicates (3)
    while not Q.empty():  # Iterate until all solutions found (4)
        # print("Murtys")
        C, _, P, S = Q.get()  # Get the top set off the Queue (4.1)
        solutions.add(tuple(S))  # Add solution to output set for searching (4.2)
        yield (
            S,
            score(P0, S),
        )  # The generator for this function returns each solution in order.
        for y, x in enumerate(
            S
        ):  # Iterating over the solution, ignoring the last (4.3)
            l = P0[y, x]  # l is the cost of the assignment y to z
            newP = P.copy()  # copy P (4.3.1)
            newP[y, x] = INF  # Remove y, z from newP (4.3.2)
            if valid(newP):  # if new solution exists (4.3.4)
                newS = hungarian(newP)  # find new solution to new problem (4.3.3)
                if tuple(newS) not in found:  # avoid duplicate answers
                    newC = score(P0, newS)  # get the cost of newS in P0
                    Q.put(
                        (newC, time(), newP, newS)
                    )  # Add (newC, newP, newS) to the queue, use time to sort equal costs
                    found.add(tuple(newS))  # Add newP, newS to found
            for i, j in np.ndindex(
                *P.shape
            ):  # From P, remove y row and z column, except for index y, z
                if (i == x and j != y) or (i != x and j == y):
                    P[j, i] = INF

    raise StopIteration()


def _hungarian(C):
    """Calculates the hungarian of the cost matrix C.
            This function performs steps 0, 1, and 2 of the munkres method.
    Complexity: O(N**3) where N is the dimensionality of C
    Time: 0.52N**3 - 2.57N**2 + 104.63N - 111.74 (ms) (See Hungarian.ods)
    Ref: http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html"""
    return _Hstep0(
        np.matrix(C).copy().T
    )  # calls step 0 on a copy of matrix C, transversed to make output correlate with assigned columns


def _Hstep0(C):
    """Create matrix such that #columns >= #rows"""
    if (
        not C.shape[1] >= C.shape[0]
    ):  # If there are not more columns than rows, traspose the matrix
        C = C.T
    return _Hstep1(C)


def _Hstep1(C):
    """For each row, find the minimum and subtract it from it's row."""
    C -= C.min(axis=1)
    return _Hstep2(C)


def _Hstep2(C):
    """For all zeroes, star zero if no other stared zero's exist in it's row or col."""
    global NONE, ZERO, STAR, PRIME
    marked = np.matrix(np.zeros(C.shape))
    ycovered, xcovered = [False for y in range(C.shape[0])], [
        False for x in range(C.shape[1])
    ]
    for y in range(C.shape[0]):
        for x in range(C.shape[1]):
            if (C[y, x] == ZERO) and not xcovered[x] and not ycovered[y]:
                marked[y, x] = STAR
                xcovered[x] = True
                ycovered[y] = True
    ycovered, xcovered = [False for y in range(C.shape[0])], [
        False for x in range(C.shape[1])
    ]

    return _Hstep3(C, marked, xcovered, ycovered)


def _Hstep3(C, marked, xcovered, ycovered):
    """Step 3:  Creates covers. Returns final answer."""
    global NONE, ZERO, STAR, PRIME
    for y in range(C.shape[0]):
        for x in range(C.shape[1]):
            if (
                marked[y, x] == STAR and not xcovered[x] and not ycovered[y]
            ):  # Cover columns which contain a star
                xcovered[x] = True
    if all(xcovered):  # If all columns are covered, we have found our answer
        return [
            findM(col, STAR)[1] for col in marked.T
        ]  # Returun the row indexes of all STARS for every column
    else:
        # print("Step 3: ",C,ycovered,xcovered,marked)
        return _Hstep4(C, marked, xcovered, ycovered)  # Otherwise, proceed to step 4


def _Hstep4(C, marked, xcovered, ycovered):
    """Step 4:  Changes the covers to cover Primes and uncover STARS.
    Primes uncovered zeros."""
    global NONE, ZERO, STAR, PRIME
    while True:
        (y, x) = findM(
            C, ZERO, xcovered, ycovered
        )  # Find first ZERO in covered Cost Matrix
        if y != NONE:
            marked[y, x] = PRIME  # Prime the uncovered zero
            if STAR not in marked[y, :]:  # If there is no STAR in the row
                # print("Step 4: ",C,ycovered,xcovered,marked)
                return _Hstep5(C, y, x, marked, xcovered, ycovered)  # Go to step 5
            else:  # Otherwise
                ycovered[y] = True  # Cover the row
                xcovered[findM(marked[y, :], STAR)[1]] = (
                    False  # Uncover the column containing the found STAR
                )
        else:
            # print("Step 4: ",C,ycovered,xcovered,marked)
            return _Hstep6(
                C, ycovered, xcovered, marked
            )  # Go to step 6 if there is no ZERO in Cost Matrix


def _Hstep5(C, y, x, marked, xcovered, ycovered):
    """Step 5:  Finds a path stairstepping from first star to first prime in star's row to first prime in star's column, etc.
    Similar to Stable Marriage Algorithm."""
    global NONE, ZERO, STAR, PRIME
    path = []
    path.append((y, x))
    while True:
        r = findM(marked[:, path[-1][1]], STAR)[0]  # Find row with a star
        if r != NONE:
            path.append(
                (r, path[-1][1])
            )  # Append row found alongside last column found
        else:
            break  # Done when none found
        c = findM(marked[path[-1][0], :], PRIME)[1]  # Find column with a prime
        path.append((path[-1][0], c))  # Append column found alongside last row found

    # Unstar each starred zero of the series, star each primed zero of the series.
    for y, x in path:
        if marked[y, x] == STAR:
            marked[y, x] = ZERO
        else:
            marked[y, x] = STAR

    # Erase all primes
    erasure = np.vectorize(lambda x: ZERO if x == PRIME else x)
    marked = erasure(marked)

    # Uncover every line in the matrix
    ycovered, xcovered = [False for y in range(C.shape[0])], [
        False for x in range(C.shape[1])
    ]

    # Return to step 3
    # print("Step 5: ",C,ycovered,xcovered,marked,path)
    return _Hstep3(C, marked, xcovered, ycovered)


def _Hstep6(C, ycovered, xcovered, marked):
    """Step 6:  Add the min value to each covered rows, and subtract it from all uncovered columns."""
    # Calculate the minimum uncovered value
    minv = uncovered(C, ycovered, xcovered)[0].min()

    # Add the min value to each covered rows, and subtract it from all uncovered columns.
    add6 = np.vectorize(lambda v, y: v + minv if ycovered[y] else v)
    sub6 = np.vectorize(lambda v, x: v - minv if not xcovered[x] else v)
    X, Y = np.meshgrid(np.arange(C.shape[1]), np.arange(C.shape[0]))
    C = add6(C, Y)
    C = sub6(C, X)

    # print("Step 6: ",C,ycovered,xcovered,marked)
    return _Hstep4(C, marked, xcovered, ycovered)


def uncovered(M, ycovered, xcovered):
    """Returns a matrix identical to M but with the covered rows and columns deleted"""
    M = M.copy()  # Copy M so that it is not directly modified
    X, Y = np.meshgrid(np.arange(M.shape[1]), np.arange(M.shape[0]))
    xcovered = [
        x for x in range(len(xcovered)) if xcovered[x]
    ]  # Enumerate indexes where x is covered
    ycovered = [
        y for y in range(len(ycovered)) if ycovered[y]
    ]  # Enumerate indexes where y is covered
    M = np.delete(M, ycovered, axis=0)  # Delete indexes where y is covered
    Y = np.delete(Y, ycovered, axis=0)
    M = np.delete(M, xcovered, axis=1)  # Delete indexes where x is covered
    X = np.delete(X, xcovered, axis=1)
    return M, Y, X


def findM(M, val, xcovered=None, ycovered=None):
    """Finds the first instance of val in M in rows and columns which are uncovered."""
    # Create default x and y covers
    if xcovered == None:
        xcovered = [False for x in range(M.shape[1])]
    if ycovered == None:
        ycovered = [False for y in range(M.shape[0])]

    # Find y and x where M[y,x] == val
    for y in range(M.shape[0]):
        for x in range(M.shape[1]):
            if not xcovered[x] and not ycovered[y]:
                if M[y, x] == val:
                    return y, x

    # Return global None if none found
    global NONE
    return NONE, NONE


# ------------ Test Methods ---------------
def timeHungarian(n2, n1=3):
    times = []
    for n in range(n1, n2):
        t = timeit(
            "hungarian(np.random.rand({0},{0}))".format(n),
            setup="from __main__ import hungarian; import numpy as np;",
        )
        times.append(t)
    return times, range(n1, n2)


def get_best_assignments(output, k_best=5):
    # Now, extract the K-best hypotheses
    ii = 0
    assignments = []
    costs = []
    for out in output:
        # Get the assignments A(row) = col, e.g., the 4th element of the assignments
        # array tells you with column of mat was selected.
        assignments.append(out[0])
        # Get the cost of the assignment.
        costs.append(out[1])
        # Print for illustration
        # print('assignment = ', assignments,', cost = ',cost)
        # In this implementation of Murty’s algorithm, we get the K-best by looping
        # over the generator and extracting the k_best assignments we are looking for.
        # Once we have done that, then break out of this loop.
        ii += 1
        if ii == k_best:
            break
    return assignments, costs


# class MurtyKBestAssigner:
#     def __init__(self, cost_matrix, k_best=10):
#         """
#         Initialize the Murty algorithm for k-best assignment problems.
#         :param cost_matrix: 2D numpy array representing the cost matrix.
#         :param k_best: Number of best solutions to find.
#         """
#         self.cost_matrix = cost_matrix
#         self.n, self.m = self.cost_matrix.shape
#         self.k_best = k_best

#     def _solve_assignment(self, cost_matrix):
#         """
#         Solve a single assignment problem using the Hungarian algorithm.
#         :param cost_matrix: 2D numpy array representing the cost matrix.
#         :return: (row_indices, col_indices), total_cost
#         """
#         row_ind, col_ind = linear_sum_assignment(cost_matrix)
#         total_cost = cost_matrix[row_ind, col_ind].sum()
#         return (row_ind, col_ind), total_cost

#     def _generate_subproblems(self, parent_solution, parent_cost_matrix):
#         """
#         Generate subproblems by excluding specific assignments from the parent solution.
#         :param parent_solution: Solution tuple (row_indices, col_indices).
#         :param parent_cost_matrix: Cost matrix of the parent problem.
#         :return: List of subproblems [(cost_matrix, lower_bound)].
#         """
#         subproblems = []
#         row_indices, col_indices = parent_solution

#         for i in range(len(row_indices)):
#             # Create a new cost matrix that excludes this assignment
#             new_cost_matrix = np.copy(parent_cost_matrix)

#             # Exclude current assignment by setting its cost to infinity
#             new_cost_matrix[row_indices[i], col_indices[i]] = np.inf

#             # Solve the subproblem
#             try:
#                 solution, cost = self._solve_assignment(new_cost_matrix)
#                 subproblems.append((new_cost_matrix, solution, cost))
#             except ValueError:
#                 # If no valid solution exists for this subproblem, skip it
#                 continue

#         return subproblems

#     def find_k_best(self):
#         """
#         Find the k-best assignment solutions using Murty's algorithm.
#         :return: List of (solution, cost) tuples.
#         """
#         # Priority queue for subproblems (min-heap based on cost)
#         priority_queue = []

#         # Solve the initial problem
#         initial_solution, initial_cost = self._solve_assignment(self.cost_matrix)
#         heapq.heappush(priority_queue, (initial_cost, self.cost_matrix, initial_solution))

#         # Store k-best solutions
#         k_best_solutions = []

#         while len(k_best_solutions) < self.k_best and priority_queue:
#             # Pop the lowest-cost subproblem from the priority queue
#             current_cost, current_cost_matrix, current_solution = heapq.heappop(priority_queue)

#             # Save this solution as one of the k-best
#             k_best_solutions.append((current_solution, current_cost))

#             # Generate subproblems from this solution and push them onto the priority queue
#             subproblems = self._generate_subproblems(current_solution, current_cost_matrix)
#             for subproblem in subproblems:
#                 new_cost_matrix, new_solution, new_cost = subproblem
#                 heapq.heappush(priority_queue, (new_cost, new_cost_matrix, new_solution))

#         return k_best_solutions


class MurtyKBestAssigner:
    def __init__(self, cost_matrix, k_best=10):
        """
        Initialize the Murty algorithm for k-best assignment problems.
        :param cost_matrix: 2D numpy array representing the cost matrix.
        :param k_best: Number of best solutions to find.
        """
        self.cost_matrix = np.array(cost_matrix)
        self.n, self.m = self.cost_matrix.shape
        self.k_best = k_best

    def _solve_assignment(self, cost_matrix):
        """
        Solve a single assignment problem using the Hungarian algorithm.
        :param cost_matrix: 2D numpy array representing the cost matrix.
        :return: (row_indices, col_indices), total_cost
        """
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_ind, col_ind].sum()
        return (row_ind, col_ind), total_cost

    def find_k_best(self):
        """
        Find the k-best assignment solutions using Murty's algorithm iteratively without a heap.
        :return: List of (solution, cost) tuples.
        """
        # List to store subproblems (sorted by cost)
        subproblems = []

        # Solve the initial problem
        initial_solution, initial_cost = self._solve_assignment(self.cost_matrix)
        subproblems.append((initial_cost, self.cost_matrix, initial_solution))

        # Store k-best solutions
        k_best_solutions = [(initial_solution, initial_cost)]

        while len(k_best_solutions) < self.k_best and subproblems:
            # Sort subproblems explicitly by cost (ascending order)
            subproblems.sort(key=lambda x: x[0])

            # Pop the lowest-cost subproblem from the list
            current_cost, current_cost_matrix, current_solution = subproblems.pop(0)

            # Save this solution as one of the k-best
            if not np.array_equal(current_solution[1], initial_solution[1]):
                k_best_solutions.append((current_solution, current_cost))

            # Generate subproblems from this solution iteratively
            row_indices, col_indices = current_solution
            for i in range(len(row_indices)):
                # Create a new cost matrix that excludes this assignment
                new_cost_matrix = np.copy(current_cost_matrix)
                new_cost_matrix[row_indices[i], col_indices[i]] = (
                    np.inf
                )  # Exclude specific assignment

                # Solve the subproblem and add it to the list if valid
                try:
                    new_solution, new_cost = self._solve_assignment(new_cost_matrix)
                    subproblems.append((new_cost, new_cost_matrix, new_solution))
                except ValueError:
                    # Skip invalid subproblems
                    continue
            initial_cost = current_cost
            initial_solution = current_solution

        return zip(*k_best_solutions)


if __name__ == "__main__":
    # Setup the matrix in Murty’s original paper to illustrate his algorithm
    mat = np.zeros((10, 10))
    # mat[0,:] = [ 7., 51., 52., 87.]
    # mat[1,:] = [ 50., 12., 0., 64.]
    # # mat[2,:] = [ 27., 77., 0., 18.]
    # # mat[3,:] = [ 62., 0., 3., 8.]

    mat[0, :] = [7.0, 51.0, 52.0, 87.0, 38.0, 60.0, 74.0, 66.0, 0.0, 20.0]
    mat[1, :] = [50.0, 12.0, 0.0, 64.0, 8.0, 53.0, 0.0, 46.0, 76, 42.0]
    mat[2, :] = [27.0, 77.0, 0.0, 18.0, 22.0, 48.0, 44.0, 13.0, 0.0, 57.0]
    mat[3, :] = [62.0, 0.0, 3.0, 8.0, 5.0, 6.0, 14.0, 0.0, 26.0, 39.0]
    mat[4, :] = [0.0, 97.0, 0.0, 5.0, 13.0, 0.0, 41.0, 31.0, 62.0, 48.0]
    mat[5, :] = [79.0, 68.0, 0.0, 0.0, 15.0, 12.0, 17.0, 47.0, 35.0, 43.0]
    mat[6, :] = [76.0, 99.0, 48.0, 27.0, 34.0, 0.0, 0.0, 0.0, 38.0, 0.0]
    mat[7, :] = [0.0, 20.0, 9.0, 27.0, 46.0, 15.0, 84.0, 19.0, 3.0, 24.0]
    mat[8, :] = [56.0, 10.0, 45.0, 39.0, 0.0, 93.0, 67.0, 79.0, 19.0, 38.0]
    mat[9, :] = [27.0, 0.0, 39.0, 53.0, 46.0, 24.0, 69.0, 46.0, 23.0, 1.0]
    # Create the generator that will produce the ranked assignments. Note that
    # this implementation of Murty’s algorithm is based on the minimum cost.
    # output = murty( mat )
    # Set the number of K-best hypotheses we want, i.e., the K value
    # Kval = 1
    # Now, extract the K-best hypotheses
    # ii = 0
    # assignments = [out[0] for out in output]
    # costs = [out[1] for out in output]

    # assignments, _ = get_best_assignments(output,5)

    # row_ind, col_ind = linear_sum_assignment(mat)
    # cost = mat[row_ind,col_ind].sum()

    # # mat[row_ind, col_ind] = 0
    # row_ind, col_ind = linear_sum_assignment(mat)
    # cost = mat[row_ind,col_ind].sum()

    assigner = MurtyKBestAssigner(mat, k_best=5)
    solutions = assigner.find_k_best()
    # mat = np.delete(mat, row_ind, axis=0)
    # mat = np.delete(mat, col_ind, axis=1)

    pass

    # for out in output :
    #     # Get the assignments A(row) = col, e.g., the 4th element of the assignments
    #     # array tells you with column of mat was selected.
    #     assignments = out[0]
    #     # Get the cost of the assignment.
    #     cost = out[1]
    #     # Print for illustration
    #     print('assignment = ', assignments,', cost = ',cost)
    #     # In this implementation of Murty’s algorithm, we get the K-best by looping
    #     # over the generator and extracting the Kval assignments we are looking for.
    #     # Once we have done that, then break out of this loop.
    #     ii += 1
    #     if ii == Kval:
    #         break
