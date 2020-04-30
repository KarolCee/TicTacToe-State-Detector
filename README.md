### How it works:
1. Load the picture.
1. After using canny edge detection, find contours of all items in the picture.
1. Divide every contour area by its bounding box area. This ratio will help to distinguish 'X' from' 'O'.
1. Delete all "duplicated" contours (if we try to detect a circle, two contours may appear, one of the inner edge of the circle, one of the outer one).
1. For each contour check if it's a board. By using Hough transform look for all straight lines in
a certain contour and their intersections. Use k-means clustering to partition all of the found intersections (points) into 4 clusters. Let's call them A B C D.
If the line A-B is almost parallel to C-D and B-C is almost parallel to A-D and A-B is almost perpendicular to B-C we can say that the contour is actually a tic tac toe board.
1. Check which contours are situated inside the found boards (thanks to coordinates of A,B,C,D we can exactly say in which part of the board).
1. The program shows a plot with the original picture, a picture with colored contours and all of the found boards shown as numpy matrices.

### How to run:
```
TicTacToe.py <path to the picture>
```
