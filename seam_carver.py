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


def _find_optimal_seam(energy, direction='vertical'):
  """use a dynamic programming approach to find the seam"""
  h, w = energy.shape
  memo = np.zeros((h, w, 2))

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

    # same as in vartical mode, work back up
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
  return ret


def _carve_seam(image, seam, direction='vertical'):
  """given an image and the seam as a list of integer
  indeces, carve the seam out and return the resultant
  image""" 
  shape  = image.shape
  nshape = list(shape)
  if direction == 'vertical':
    # if we are trimming vertically (a column) then the
    # output image will have one fewer column
    nshape[1] -= 1
    nshape = tuple(nshape)
    ret = np.zeros(nshape, image.dtype)
    
    # now we copy and skip the column on the seam
    for y in range(shape[0]):
      x = 0
      for _x in range(shape[1]):
        if _x == seam[y]:
          continue
        ret[y, x] = image[y, _x]
        x += 1

  # same deal here, except rows instead of columns
  else:
    nshape[0] -= 1
    nshape = tuple(nshape)
    ret = np.zeros(nshape, image.dtype)
    
    for x in range(shape[1]):
      y = 0
      for _y in range(shape[0]):
        if _y == seam[x]:
          continue
        ret[y, x] = image[_y, x]
        y += 1

  return ret


def seam_carver(image, aspect_ratio):
  """given an image and the desired aspect ratio, use 
  seam carving to retarget the image to fit the desired
  ratio"""
  global _ksize
  image   = image.copy()
  h, w, c = image.shape

  # get sgm
  dx  = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=_ksize)
  dy  = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=_ksize)
  sgm = np.zeros((h, w), dtype=np.float64)
  m   = 0
  for x in range(w):
    for y in range(h):
      for i in range(c):
        sgm[y, x] += (dx[y, x, i]*dx[y, x, i] + dy[y, x, i]*dy[y, x, i])
      sgm[y, x] = sqrt(sgm[y, x])
      m = max(m, sgm[y, x])
  
  # determine resizing strategy
  old_aspect_ratio = w / h

  # if the aspect ratio increased, then we should 
  # remove horizontal seams from the image
  if old_aspect_ratio < aspect_ratio:
    # let's find the number of rows to remove
    remove = int(h - w/aspect_ratio)

    # now we can iterate over the number of seams to remove
    for _ in range(remove):
      seam  = _find_optimal_seam(sgm,   direction='horizontal')
      image = _carve_seam(image, seam, direction='horizontal')
      sgm   = _carve_seam(sgm,   seam, direction='horizontal')
    
  # if the aspect ratio decreased, then we should
  # remove vertical seams from the image
  elif old_aspect_ratio > aspect_ratio:
    remove = int(w - aspect_ratio*h)

    # now we can iterate over the number of seams to remove
    for _ in range(remove):
      seam  = _find_optimal_seam(sgm,   direction='vertical')
      image = _carve_seam(image, seam, direction='vertical')
      sgm   = _carve_seam(sgm,   seam, direction='vertical')
        
  return image

def main(argv) -> int:
  if len(argv) != 4:
    print('Usage: ' + argv[0] + ' <input_image> <output_image> <ratio>')
    return 1

  pic = cv.imread('dog.jpg')

  if pic is None:
    print('File ' + argv[1] + ' couldn\'t be found')
    return 1

  resize = seam_carver(pic, float(argv[3]))
  cv.imwrite(argv[2], resize)
  return 0

if __name__ == '__main__':
  import sys
  main(sys.argv)
