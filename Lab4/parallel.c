/*
 *  Name: Daniel Wu
 *  UID: 706382792
 */

#include <stdlib.h>
#include <omp.h>

#include "utils.h"
#include "parallel.h"

/*
 *  PHASE 1: compute the mean pixel value
 *  This code is buggy! Find the bug and speed it up.
 */
void mean_pixel_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, double mean[NUM_CHANNELS])
{
    int row, col, ch;
    long count;

    count = (long)num_rows * num_cols;

    if (count == 0) {
        for (ch = 0; ch < NUM_CHANNELS; ch++) {
            mean[ch] = 0.0;
        }
        return;
    }

    double temp_sum_ch0 = 0.0;
    double temp_sum_ch1 = 0.0;
    double temp_sum_ch2 = 0.0;

    #pragma omp parallel for private(col) reduction(+:temp_sum_ch0, temp_sum_ch1, temp_sum_ch2)
    for (row = 0; row < num_rows; row++) {
        for (col = 0; col < num_cols; col++) {
            temp_sum_ch0 += img[row * num_cols + col][0];
            temp_sum_ch1 += img[row * num_cols + col][1];
            temp_sum_ch2 += img[row * num_cols + col][2];
        }
    }

    mean[0] = temp_sum_ch0;
    mean[1] = temp_sum_ch1;
    mean[2] = temp_sum_ch2;

    for (ch = 0; ch < NUM_CHANNELS; ch++) {
        mean[ch] /= count;
    }

}

/*
 *  PHASE 2: convert image to grayscale and record the max grayscale value along with the number of times it appears
 *  This code is NOT buggy, just sequential. Speed it up.
 */
void grayscale_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, uint32_t grayscale_img[][NUM_CHANNELS], uint8_t *max_gray, uint32_t *max_count)
{
    int row, col, ch, gray_ch;
    *max_gray = 0;
    *max_count = 0;

    #pragma omp parallel for schedule(static) collapse(2)
    for (row = 0; row < num_rows; row++) {
        for (col = 0; col < num_cols; col++) {
            uint32_t gray_value = 0;
            for (ch = 0; ch < NUM_CHANNELS; ch++) {
                gray_value += img[row * num_cols + col][ch];
            }
            gray_value /= NUM_CHANNELS;

            for (gray_ch = 0; gray_ch < NUM_CHANNELS; gray_ch++) {
                grayscale_img[row * num_cols + col][gray_ch] = gray_value;
            }
        }
    }

    uint8_t local_max = 0;
    uint32_t local_count = 0;
    
    #pragma omp parallel
    {
        uint8_t thread_max = 0;
        uint32_t thread_count = 0;
        
        #pragma omp for schedule(static) collapse(2)
        for (row = 0; row < num_rows; row++) {
            for (col = 0; col < num_cols; col++) {
                uint8_t pixel_gray = grayscale_img[row * num_cols + col][0];
                
                if (pixel_gray > thread_max) {
                    thread_max = pixel_gray;
                    thread_count = NUM_CHANNELS;
                } else if (pixel_gray == thread_max) {
                    thread_count += NUM_CHANNELS;
                }
            }
        }
        
        #pragma omp critical
        {
            if (thread_max > local_max) {
                local_max = thread_max;
                local_count = thread_count;
            } else if (thread_max == local_max) {
                local_count += thread_count;
            }
        }
    }
    
    *max_gray = local_max;
    *max_count = local_count;
}

/*
 *  PHASE 3: perform convolution on image
 *  This code is NOT buggy, just sequential. Speed it up.
 */
void convolution_parallel(const uint8_t padded_img[][NUM_CHANNELS], int num_rows, int num_cols, const uint32_t kernel[], int kernel_size, uint32_t convolved_img[][NUM_CHANNELS])
{
    int row, col, ch, kernel_row, kernel_col;
    int kernel_norm, i;
    int conv_rows, conv_cols;

    kernel_norm = 0;
    for (i = 0; i < kernel_size * kernel_size; i++) {
        kernel_norm += kernel[i];
    }

    conv_rows = num_rows - kernel_size + 1;
    conv_cols = num_cols - kernel_size + 1;

    #pragma omp parallel for schedule(static) collapse(2)
    for (row = 0; row < conv_rows; row++) {
        for (col = 0; col < conv_cols; col++) {
            for (ch = 0; ch < NUM_CHANNELS; ch++) {
                uint32_t sum = 0;
                
                for (kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                    for (kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                        sum += padded_img[(row + kernel_row) * num_cols + col + kernel_col][ch] * 
                               kernel[kernel_row * kernel_size + kernel_col];
                    }
                }
                
                convolved_img[row * conv_cols + col][ch] = sum / kernel_norm;
            }
        }
    }
}