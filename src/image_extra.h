/*
 * image_extra.h
 *
 *  Created on: Mar 17, 2016
 *      Author: "Michal.Busta at gmail.com"
 */

#ifndef IMAGE_EXTRA_H_
#define IMAGE_EXTRA_H_


#ifdef OPENCV
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>


image get_image_from_stream(CvCapture *cap);
image ipl_to_image(IplImage* src);

#endif


#endif /* IMAGE_EXTRA_H_ */
