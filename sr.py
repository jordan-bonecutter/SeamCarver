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
  if left < middle:
    return (left + energy, -1)
  else:
    return (middle + energy, 0)


def _optimal_seamr(middle, right, energy):
  if middle < right:
    return (middle + energy, 0)
  else:
    return (right + energy, 1)


def _optimal_seam3(left, middle, right, energy):
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

  #energy = np.zeros((5, 6))
  #energy[0] = 240.18, 225.59, 302.27, 159.43, 181.81, 192.99
  #energy[1] = 124.18, 237.35, 151.02, 234.09, 107.89, 159.67
  #energy[2] = 111.10, 138.69, 228.10, 133.07, 211.51, 143.75
  #energy[3] = 130.67, 153.88, 174.01, 284.01, 194.50, 213.53
  #energy[4] = 179.82, 175.49,  70.06, 270.80, 201.53, 191.20

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

  ret.reverse()
  return ret


def _remove_seam(image, seam, direction='vertical'):
  shape  = image.shape
  nshape = list(shape)
  if direction == 'vertical':
    nshape[1] -= 1
    nshape = tuple(nshape)
    ret = np.zeros(nshape, image.dtype)
    
    for y in range(shape[0]):
      x = 0
      for _x in range(shape[1]):
        if _x == seam[y]:
          continue
        ret[y, x] = image[y, _x]
        x += 1

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


def seam_removal(image, aspect_ratio):
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
      image = _remove_seam(image, seam, direction='horizontal')
      sgm   = _remove_seam(sgm,   seam, direction='horizontal')
      print((_+1)/(remove))
    
  # if the aspect ratio decreased, then we should
  # remove vertical seams from the image
  elif old_aspect_ratio > aspect_ratio:
    remove = int(w - aspect_ratio*h)

    # now we can iterate over the number of seams to remove
    for _ in range(remove):
      seam  = _find_optimal_seam(sgm,   direction='vertical')
      image = _remove_seam(image, seam, direction='vertical')
      sgm   = _remove_seam(sgm,   seam, direction='vertical')
      print((_+1)/(remove))
        
  return image

def main() -> int:
  pic = cv.imread('dog.jpg')
  resize = seam_removal(pic, 1.75)
  cv.imwrite('dognew.jpg', resize)
  return 0

if __name__ == '__main__':
  main()
