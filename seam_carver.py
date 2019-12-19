#
#
#
#

import cv2   as cv
import numpy as np
from math import sqrt
from math import log

_ksize = 3


def _optimal_seaml(left, middle, energy):
  """find the optimal path given two choices (no right)"""
  if left < middle:
    return (left + energy, -1)
  else:
    return (middle + energy, 0)


def _optimal_seamr(middle, right, energy):
  """find the optimal path given two choices (no left)"""
  if middle < right:
    return (middle + energy, 0)
  else:
    return (right + energy, 1)


def _optimal_seam3(left, middle, right, energy):
  """find the optimal path given three choices"""
  if left < middle:
    if left < right:
      return (left + energy, -1)
    else:
      return (right + energy, 1)
  elif right < middle:
    return (right + energy, 1)
  else:
    return (middle + energy, 0)


def _find_optimal_seam(energy, shape, direction='vertical'):
  """use a dynamic programming approach to find the seam.
  energy: np.array(h, w)
  direction: 'vertical' or 'horizontal'
  return: seam: list of integers corresponding to indeces to be removed"""
  h, w = tuple(shape)
  memo = np.zeros((h, w, 2))

  # this approach to seam carving utilizes dp as mentioned
  # in the paper. essentially, we want to find the minimum 
  # cost from the top row to the bottom row. we can treat 
  # the problem in stages. first, we look at every pixel in
  # the top row and set it's cost equal to the energy at 
  # that pixel. next we add on the next row. if we want to 
  # find the cost of a path for a pixel in the second row,
  # we can say that it is equal to the minimum of the cost 
  # of the three adjacent paths above it plus it's own 
  # energy. in other words: 
  # memo[y, x] = min(memo[y-1,x-1], memo[y-1,x], memo[y-1,x+1]) + energy[y, x]
  # and
  # memo[0, x] = energy[0, x]
  # so we continue this process all the way down the image.
  # once we get to the bottom row, we work our way back up.
  # we also record the steps we take along the way so it's
  # easier to work back up at the end
  if direction == 'vertical':
    # the first row in the memoize array is the
    # first row of the energy array
    for x in range(w):
      memo[0, x] = (energy[0, x], 0)

    # now we start the dp approach
    for y in range(1, h):
      memo[y, 0] = _optimal_seamr(memo[y-1, 0,0], memo[y-1, 1,0], energy[y, 0])
      for x in range(1, w-1):
        memo[y, x] = _optimal_seam3(memo[y-1, x-1,0], memo[y-1, x,0], memo[y-1, x+1,0], energy[y, x])
      memo[y, w-1] = _optimal_seaml(memo[y-1, w-2,0], memo[y-1, w-1,0], energy[y, w-1])

    # once we have the memoized array, we can 
    # work backwards from the bottom by finding the
    # minimum value and retracing our path from there
    ret = []
    min_index = 0
    min_val   = memo[h-1, w-1,0] + 1
    for x in range(w):
      if memo[h-1, x,0] < min_val:
        min_val   = memo[h-1, x,0]
        min_index = x
    
    curr = min_index
    for y in range(h-1):
      ret.append(curr)  
      curr += int(memo[h-1-y, curr, 1])
    ret.append(curr)

  else:
    # the first row in the memoize array is the
    # first row of the energy array
    for y in range(h):
      memo[y, 0] = (energy[y, 0], 0)

    # now we start the dp approach
    for x in range(1, w):
      memo[0, x] = _optimal_seamr(memo[0, x-1, 0], memo[1, x-1, 0], energy[0, x])
      for y in range(1, h-1):
        memo[y, x] = _optimal_seam3(memo[y-1, x-1,0], memo[y, x-1,0], memo[y+1, x-1,0], energy[y, x])
      memo[h-1, x] = _optimal_seaml(memo[h-2, x-1,0], memo[h-1, x-1,0], energy[h-1, x])

    # same as in vertical mode, work back up
    # from the bottom of the memoized array
    ret = []
    min_index = 0
    min_val   = memo[h-1, w-1,0] + 1
    for y in range(h):
      if memo[y, w-1,0] < min_val:
        min_val   = memo[y, w-1,0]
        min_index = y
    
    curr = min_index
    for x in range(w-1):
      ret.append(curr)  
      curr += int(memo[curr, w-1-x, 1])
    ret.append(curr)

  # reverse the array so it's easier to use
  # cause python
  ret.reverse()

  # format of return value: [1, 2, 2, 3, 2, ...] would mean
  # to remove the pixel in column 1 in the row 0, column 2
  # in row 1, column 2 in row 2, column 3 in row 3, column
  # 2 in row 4, ..., column c_i in row h-1 (vertical mode).
  #
  # in horizontal mode, it means much the same except flip
  # the rows and the columns
  return ret


def _carve_seam(image, shape, seam, direction='vertical'):
  """given an image and the seam as a list of integer
  indeces, carve the seam out and return the resultant
  image""" 
  if direction == 'vertical':
    # if we are trimming vertically (a column) then the
    # output image will have one fewer column
    shape[1] -= 1
    
    # now we copy and skip the column on the seam
    for y in range(shape[0]):
      for x in range(seam[y], shape[1]):
        image[y, x] = image[y, x+1]

  # same deal here, except rows instead of columns
  else:
    shape[0] -= 1
    
    for x in range(shape[1]):
      for y in range(seam[x], shape[0]):
        image[y, x] = image[y+1, x]


def seam_carver(image, aspect_ratio):
  """given an image and the desired aspect ratio, use 
  seam carving to retarget the image to fit the desired
  ratio.
  strategy = carve or add"""
  global _ksize
  image   = image.copy()
  h, w, c = image.shape

  # get sgm
  dx  = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=_ksize)
  dy  = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=_ksize)
  sgm = np.zeros((h, w), dtype=np.float64)
  for x in range(w):
    for y in range(h):
      for i in range(c):
        sgm[y, x] += (dx[y, x, i]*dx[y, x, i] + dy[y, x, i]*dy[y, x, i])
      sgm[y, x] = sqrt(sgm[y, x])
  
  # determine resizing strategy
  old_aspect_ratio = w / h

  # if the aspect ratio increased, then we should 
  # remove horizontal seams from the image
  if old_aspect_ratio < aspect_ratio:
    # let's find the number of rows to remove
    remove = int(h - w/aspect_ratio)
    ishape = [h, w, c]
    eshape = [h, w]

    # now we can iterate over the number of seams to remove
    for _ in range(remove):
      seam = _find_optimal_seam(sgm, eshape, direction='horizontal')
      _carve_seam(image, ishape, seam, direction='horizontal')
      _carve_seam(sgm,   eshape, seam, direction='horizontal')

    image = image[:h-remove, :]
    
  # if the aspect ratio decreased, then we should
  # remove vertical seams from the image
  elif old_aspect_ratio > aspect_ratio:
    remove = int(w - aspect_ratio*h)
    ishape = [h, w, c]
    eshape = [h, w]

    # now we can iterate over the number of seams to remove
    for _ in range(remove):
      seam = _find_optimal_seam(sgm, eshape, direction='vertical')
      _carve_seam(image, ishape, seam, direction='vertical')
      _carve_seam(sgm,   eshape, seam, direction='vertical')

    image = image[:, :w-remove]
        
  return image

def main(argv) -> int:
  if len(argv) != 4:
    print('Usage: ' + argv[0] + ' <input_image> <output_image> <ratio>')
    return 1

  pic = cv.imread(argv[1])

  if pic is None:
    print('File ' + argv[1] + ' couldn\'t be found')
    return 1

  resize = seam_carver(pic, float(argv[3]))
  cv.imwrite(argv[2], resize)
  return 0

if __name__ == '__main__':
  import sys
  main(sys.argv)
