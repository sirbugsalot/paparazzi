#include "modules/computer_vision/cv_detect_color_object_custom.h"
#include "modules/computer_vision/cv.h"
#include "modules/core/abi.h"
#include "std.h"

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"
#include "state.h"


#define PRINT(string,...) fprintf(stderr, "[object_detector->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if OBJECT_DETECTOR_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static pthread_mutex_t mutex;

#ifndef COLOR_OBJECT_DETECTOR_FPS1
#define COLOR_OBJECT_DETECTOR_FPS1 0 ///< Default FPS (zero means run at camera fps)
#endif


//NEW GLOBAL VARIABLES

// Kernel size for constructing vectors
#ifndef kernel_size
#define kernel_size 15
#endif

// Manually insert the following parameters
#ifndef half_kernel_size
#define half_kernel_size 7
#endif

// floor(520 / kernel_size)
#ifndef vector_array_length
#define vector_array_length 34
#endif

// floor(vector_array_length / 2)
#ifndef vector_array_mid
#define vector_array_mid 17
#endif

// in_nps = 1 mean true (running in simulator or real life)
#ifndef in_nps
#define in_nps 0
#endif

// curtain_open = 1 means true
#ifndef curtain_open
#define curtain_open 1
#endif


float float_angle_norm(float a) {
  while (a > M_PI)
  {
    a -= (2.*M_PI);
  }
  while (a < M_PI)
  {
    a += (2.*M_PI);
  }
  return a;  
}

// Filter Settings
uint8_t cod_lum_min1 = 0;
uint8_t cod_lum_max1 = 0;
uint8_t cod_cb_min1 = 0;
uint8_t cod_cb_max1 = 0;
uint8_t cod_cr_min1 = 0;
uint8_t cod_cr_max1 = 0;

uint8_t cod_lum_min2 = 0;
uint8_t cod_lum_max2 = 0;
uint8_t cod_cb_min2 = 0;
uint8_t cod_cb_max2 = 0;
uint8_t cod_cr_min2 = 0;
uint8_t cod_cr_max2 = 0;

bool cod_draw1 = false;
bool cod_draw2 = false;

// define Structs

// Struct for information to be passed on in the abi message
struct color_object_t {
  int32_t x_c;
  int32_t y_c;
  uint32_t color_count;
  bool updated; 

  int16_t vector_x;
  int16_t vector_y;
  int32_t direction;
};

// Struct for output of find_object_centroid function
struct return_value {
  uint32_t color_count;
  int16_t vector_x;
  int16_t vector_y;
  int32_t direction;
};

// Struct for YUV pointers
struct pixel_values {
  uint8_t *yp;
  uint8_t *up;
  uint8_t *vp;
};

struct color_object_t global_filters[1];

struct return_value find_object_centroid(struct image_t *img, int32_t* p_xc, int32_t* p_yc, bool draw,
                              uint8_t lum_min, uint8_t lum_max,
                              uint8_t cb_min, uint8_t cb_max,
                              uint8_t cr_min, uint8_t cr_max);


static struct image_t *object_detector(struct image_t *img, uint8_t filter)
{
  uint8_t lum_min, lum_max;
  uint8_t cb_min, cb_max;
  uint8_t cr_min, cr_max;
  bool draw;

  switch (filter){
    case 1:
      lum_min = cod_lum_min1;
      lum_max = cod_lum_max1;
      cb_min = cod_cb_min1;
      cb_max = cod_cb_max1;
      cr_min = cod_cr_min1;
      cr_max = cod_cr_max1;
      draw = cod_draw1;
      break;
    default:
      return img;
  };

  int32_t x_c, y_c;

  struct return_value result;
  result = find_object_centroid(img, &x_c, &y_c, draw, lum_min, lum_max, cb_min, cb_max, cr_min, cr_max);
  VERBOSE_PRINT("Color count %d: %u, threshold %u, x_c %d, y_c %d\n", camera, object_count, count_threshold, x_c, y_c);
  VERBOSE_PRINT("centroid %d: (%d, %d) r: %4.2f a: %4.2f\n", camera, x_c, y_c,
        hypotf(x_c, y_c) / hypotf(img->w * 0.5, img->h * 0.5), RadOfDeg(atan2f(y_c, x_c)));

  pthread_mutex_lock(&mutex);
  global_filters[0].color_count = result.color_count;
  global_filters[0].x_c = x_c;
  global_filters[0].y_c = y_c;
  global_filters[0].updated = true;
  global_filters[0].vector_x = result.vector_x;
  global_filters[0].vector_y = result.vector_y;  
  global_filters[0].direction = result.direction;
  pthread_mutex_unlock(&mutex);

  return img;
}

struct image_t *object_detector1(struct image_t *img, uint8_t camera_id);
struct image_t *object_detector1(struct image_t *img, uint8_t camera_id __attribute__((unused)))
{
  return object_detector(img, 1);
}

void color_object_detector_init(void)
{
  memset(global_filters, 0, sizeof(struct color_object_t));
  pthread_mutex_init(&mutex, NULL);
#ifdef COLOR_OBJECT_DETECTOR_CAMERA1
#ifdef COLOR_OBJECT_DETECTOR_LUM_MIN1
  cod_lum_min1 = COLOR_OBJECT_DETECTOR_LUM_MIN1;
  cod_lum_max1 = COLOR_OBJECT_DETECTOR_LUM_MAX1;
  cod_cb_min1 = COLOR_OBJECT_DETECTOR_CB_MIN1;
  cod_cb_max1 = COLOR_OBJECT_DETECTOR_CB_MAX1;
  cod_cr_min1 = COLOR_OBJECT_DETECTOR_CR_MIN1;
  cod_cr_max1 = COLOR_OBJECT_DETECTOR_CR_MAX1;
#endif
#ifdef COLOR_OBJECT_DETECTOR_DRAW1
  cod_draw1 = COLOR_OBJECT_DETECTOR_DRAW1;
#endif

  cv_add_to_device(&COLOR_OBJECT_DETECTOR_CAMERA1, object_detector1, COLOR_OBJECT_DETECTOR_FPS1, 0);
#endif
}

// Function to return YUV pointers for a given x and y coordinate pixel on the image
struct pixel_values compute_pixel_yuv(struct image_t *img, int16_t x, int16_t y)
{
  struct pixel_values result;
  uint8_t *buffer = img->buf;
  uint8_t *yp, *up, *vp;
  if (x % 2 == 0) {
    // Even x
    up = &buffer[y * 2 * img->w + 2 * x];      // U
    yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y1
    vp = &buffer[y * 2 * img->w + 2 * x + 2];  // V
    //yp = &buffer[y * 2 * img->w + 2 * x + 3]; // Y2
  } else {
    // Uneven x
    up = &buffer[y * 2 * img->w + 2 * x - 2];  // U
    //yp = &buffer[y * 2 * img->w + 2 * x - 1]; // Y1
    vp = &buffer[y * 2 * img->w + 2 * x];      // V
    yp = &buffer[y * 2 * img->w + 2 * x + 1];  // Y2
  }
  result.yp = yp;
  result.up = up;
  result.vp = vp;
  return result;
}

int* generate_sub_mask(struct image_t *img, int target_h, int target_w, int og_h, int og_w, bool draw, uint8_t lum_min, uint8_t lum_max,
                              uint8_t cb_min, uint8_t cb_max,
                              uint8_t cr_min, uint8_t cr_max){
  // int row = 0;
  int * mask_array = malloc(sizeof(int)*target_h*target_w);
  for (int el=0; el<target_h*target_w; el++){
    mask_array[el] = 0;
  }
  // int * mask_array[target_h*target_w] = {0};
  int counter = 0;
  struct pixel_values pix_values; 

  for (int j = target_h; j >=0; j--)
  // for (int j = og_h-target_h; j <og_h; j++)
  {
    for (int i = (og_w-target_w)/2; i < (og_w+target_w)/2; i++)
    {
      uint8_t *yp, *up, *vp;
      pix_values = compute_pixel_yuv(img, j, i);
      yp = pix_values.yp;
      up = pix_values.up;
      vp = pix_values.vp;

      // *yp = 2*j;
      // *up = 2*i;

      // if ((i ==(og_w-target_w)/2) || (i ==(og_w-target_w)/2)){
      //   *vp = 255;
      //   *up = 255;
      // }
      if ( (*yp >= lum_min) && (*yp <= lum_max) &&
        (*up >= cb_min ) && (*up <= cb_max ) &&
        (*vp >= cr_min ) && (*vp <= cr_max )) {
          
        mask_array[counter] = 1;
        if (draw){
                *vp = 255;
                // *up = 255;
                // *up = 255; 
        }
        }
      // else{
      //   mask_array[counter] = 0;
      //   if (draw){
      //           // *vp = 0; 
      //           *up = 255;
      //           *vp = 255;
      //           *yp = 255;
      //   }
      // }
        counter ++;

}
    
  }
  return mask_array;
  // return img;
  
}


int find_next(int im_bw[], int prev_points[],  const int h, const int w, int step, int nr_prev_points){
    int* init_guess = malloc(2 * sizeof(int));
    int* dir = malloc(2 * sizeof(int));
    // int* init_guess[2] = {0};
    // int* dir[2] = {0};

    // int nr_prev_points = sizeof(prev_points)/sizeof(int);
    
    // malloc((2+nr_prev_points)*sizeof(int))
    int* points_arr = malloc((2+nr_prev_points)*sizeof(int));
    points_arr[0] = prev_points[0];
    points_arr[1] = prev_points[0];
    // printf("\n");
    // printf("\nPoint in prevPoints: %d",points_arr[0]);
    // printf("\nPoint in prevPoints: %d",points_arr[1]);
    for (int i = 0; i < nr_prev_points; i++)
    {
        points_arr[2+i] = prev_points[i];
        // printf("\nPoint in prevPoints: %d",prev_points[i]);
    }
    
    // printf("\n");

    int prev_len = nr_prev_points+2;
    

    dir[0] = ((points_arr[prev_len-3]*1+points_arr[prev_len-2]*2+points_arr[prev_len-1]*3)/6) - points_arr[prev_len-2];
    dir[1] = 3 * (prev_len-2);
    // printf("\nDirection in y: %d",dir[0]);
    init_guess[0] = points_arr[prev_len-1] + dir[0];
    init_guess[1] = dir[1];
    // printf("\nINIT GUESS: %d, %d",init_guess[0],init_guess[1]);
    int edge_found = 0;
    int ij = 0;
    while (!edge_found && ij < 100) {
        ij++;
        if (init_guess[0] + 1 < h) {
            if ((im_bw[init_guess[0]*w + init_guess[1]] + im_bw[(init_guess[0]-1)*w + init_guess[1]]) == 1) {
                edge_found = 1;
                }
            else if ((im_bw[init_guess[0]*w + init_guess[1]] + im_bw[(init_guess[0]-1)*w + init_guess[1]]) == 0)
            {
                init_guess[0] ++;
            }
            else{
                init_guess[0] --;
            }
            }

        else{
            return h;
        }

    }
    return init_guess[0];
}

int* hor_tracer(int sub_im[], int step, int h, int w,struct image_t *img){
    // step = 5;
    struct pixel_values pix_values; 
    bool ground_found = false;
    int i = 0;

    int col = 0;
    int area = 0;

    printf("\nSubwindow h: %d, w: %d", h,w);
    /* find first point*/
    while (ground_found==false && i < (h-5) && col < (w-5))
    {
        if (sub_im[i*w + col] != 0){
           area = 0; 
           for (int j = 0; j < 5; j++)
           {
            for (int k = 0; k < 5; k++)
            {
                area = area + sub_im[(i+j)*w + col + k];
                // uint8_t *yp, *up, *vp;
                // pix_values = compute_pixel_yuv(img, 520/2-w/2+ i+j, col+k);
                // yp = pix_values.yp;
                // up = pix_values.up;
                // vp = pix_values.vp;

                // *yp = 255;
                // *up = 255;
                // *vp = 255;
            }
            
           }
           if (area>=20)
            {
                ground_found = true;
            }
            
           
        }
      //   uint8_t *yp, *up, *vp;
      // pix_values = compute_pixel_yuv(img, 520/2-w/2+ i, col+k);
      // yp = pix_values.yp;
      // up = pix_values.up;
      // vp = pix_values.vp;

      // *yp = 255;
      // *up = 255;
      // *vp = 255;
        i++;
        if (i==h-5)
        {
            i = 0;
            col++;
        }
    }
    int start_row = i;
    // uint8_t *yp, *up, *vp;
      // pix_values = compute_pixel_yuv(img, 520/2-w/2+ i+j, col+k);
      // yp = pix_values.yp;
      // up = pix_values.up;
      // vp = pix_values.vp;

      // *yp = 255;
      // *up = 255;
      // *vp = 255;
    printf("\nstart row: %d", start_row);
    // printf("\nstart point: %d", start_row);

    bool edge_dir_found = false;
    int * hor_points_y = malloc(sizeof(int)*w/step);
    // int * hor_points_y[w/step] = {0};
    hor_points_y[0] = start_row;

    i = 0;
    // bool end = false;
    // printf("\nWidth: %d, step: %d", w, step);
    printf("\nNumber of points to be found: %d", 1+w/step);
    for (int index = 1; index < w/step+1; index++)  
    {
        int next_y  = find_next(sub_im, hor_points_y, h,w,step, index);
        hor_points_y[index] = next_y;
        

        printf("\nNext y point: %d", hor_points_y[index]);        
    }
    return hor_points_y;
        
    }
    
int mom_changes(int hor_points_y[],int length, float thr, int step){
    // f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h^2
    // note we are changing the original hor_points array if we change hor_points_y
    int count = 0;
    // printf("\nlength: %d", length);

    for (int i = 1; i < length-1; i++)
    {
        float fdd = (float)((hor_points_y)[i-1] - (hor_points_y[i])*2 + hor_points_y[i+1])/(step*step);
        // printf("\nhorizon points: %d", ((hor_points_y)[i-1] - (hor_points_y[i])*2 + hor_points_y[i+1])/step);
        // printf("\nfloat: %f", fdd);
        if (fdd>thr || -fdd>thr)
        {
            count++;
            // printf(count);
        }   
    } 
    return count;}

int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

int safety(int hor_points_y[], int length, int h){
    // CALL THIS FUNCTION AS LAST ONE CAUSE IT CHANGES ORDER OF THE ELEMENTS!!!!!!!
    qsort(hor_points_y, length, sizeof(int),cmpfunc);

    int avg = (hor_points_y[0] + hor_points_y[1] + hor_points_y[2])/3;
    
    int safety = (int)((float)(h-avg)/h*240);
    return safety;
}

// Main visual processing and classification function
struct return_value find_object_centroid(struct image_t *img, int32_t* p_xc, int32_t* p_yc, bool draw,
                              uint8_t lum_min, uint8_t lum_max,
                              uint8_t cb_min, uint8_t cb_max,
                              uint8_t cr_min, uint8_t cr_max)
{

  // Initialize local variables
  uint32_t cnt = 0;
  uint32_t tot_x = 0;
  uint32_t tot_y = 0;
  uint8_t *buffer = img->buf;

  uint8_t mask_array[240][520] = {0};

  struct return_value test;
  struct pixel_values pix_values;  
  
  int16_t heigth = img->h;
  int16_t width = img->w;
  int16_t kernel_cnt = 0;
  // Kernel treshold on pixel count for drawing vectors
  int16_t threshold = 180;

  int16_t x = 0;
  int16_t y = 0;
  int16_t kur_x = 0;
  int16_t kur_y = 0;

  int16_t kernel_centroid = 0;

  int16_t kernel_w_cnt = floor(floor(width/kernel_size)*0.5); // Multiply by 0.5 such that only the bottom half of the image is scanned
  int16_t kernel_h_cnt = floor(heigth/kernel_size);

  int16_t vector_array[vector_array_length] = {0};

  // Loop over y coordinates of the image (per kernel block)
  for (int8_t y_k = 0; y_k < kernel_h_cnt; y_k++){

      int8_t state = 0;

      // Loop over the x coordinate of the image, only bottom half (see above)
      for (int8_t x_k = kernel_w_cnt-1; x_k >= 0; x_k--){
      
      kernel_cnt = 0;
      kur_x = kernel_size*x_k;
      kur_y = kernel_size*y_k;
      // Loop through the x and y coordinate in the selected kernel
      for (int8_t i = 0; i < kernel_size; i++){
        for (int8_t j = 0; j < kernel_size; j++){
          x = kur_x + j;
          y = kur_y + i;

          // Retrieve YUV values
          uint8_t *yp, *up, *vp;
          pix_values = compute_pixel_yuv(img, x, y);
          yp = pix_values.yp;
          up = pix_values.up;
          vp = pix_values.vp;

          // Classify pixel
          if (in_nps){ // Filter for simulator (simple)
            if ( (*yp >= lum_min) && (*yp <= lum_max) &&
              (*up >= cb_min ) && (*up <= cb_max ) &&
              (*vp >= cr_min ) && (*vp <= cr_max )) {
              if (draw){
                *yp = 255;  // make pixel brighter in image
              }
              kernel_cnt++;
              mask_array[x][y] = 1;
              }
          }

          else {

            if (curtain_open) {
							if( (*up <= 108.5) && (*vp <= 144.5) && (*yp > 124.5) && (*yp <= 186.5) && (*yp > 134.5) && (*vp > 128.5) ){
								if (draw){
									*yp = 255;  // make pixel brighter in image
								}
								kernel_cnt++;
							}
							if( (*up <= 108.5) && (*vp > 144.5) && (*vp <= 150.5) && (*yp <= 184.5) && (*yp > 133.5) && (*up <= 102.5) ){
								if (draw){
									*yp = 255;  // make pixel brighter in image
								}
								kernel_cnt++;
							}
							if( (*up > 108.5) && (*up <= 111.5) && (*vp <= 143.5) && (*yp > 135.5) && (*yp <= 184.5) && (*yp > 143.5) ){
								if (draw){
									*yp = 255;  // make pixel brighter in image
								}
								kernel_cnt++;
							}
							if( (*up <= 108.5) && (*vp <= 144.5) && (*yp > 124.5) && (*yp <= 186.5) && (*yp <= 134.5) && (*up <= 104.5) ){
								if (draw){
									*yp = 255;  // make pixel brighter in image
								}
								kernel_cnt++;
							}
							if( (*up <= 108.5) && (*vp <= 144.5) && (*yp > 124.5) && (*yp <= 186.5) && (*yp > 134.5) && (*vp <= 128.5) ){
								if (draw){
									*yp = 255;  // make pixel brighter in image
								}
								kernel_cnt++;
							}
							if( (*up <= 108.5) && (*vp <= 144.5) && (*yp <= 124.5) && (*yp > 112.5) && (*up <= 103.5) && (*yp > 120.5) ){
								if (draw){
									*yp = 255;  // make pixel brighter in image
								}
								kernel_cnt++;
							}

            }
            else {
							if( (*up <= 107.5) && (*vp <= 144.5) && (*yp > 101.5) && (*yp <= 165.5) && (*vp <= 141.5) && (*up > 84.5) ){
								if (draw){
									*yp = 255;  // make pixel brighter in image
								}
								kernel_cnt++;
							}
							if( (*up <= 107.5) && (*vp <= 144.5) && (*yp > 101.5) && (*yp <= 165.5) && (*vp > 141.5) && (*yp > 111.5) ){
								if (draw){
									*yp = 255;  // make pixel brighter in image
								}
								kernel_cnt++;
							}
							if( (*up > 107.5) && (*up <= 110.5) && (*vp <= 141.5) && (*yp > 101.5) && (*yp <= 163.5) && (*yp > 111.5) ){
								if (draw){
									*yp = 255;  // make pixel brighter in image
								}
								kernel_cnt++;
							}
							if( (*up <= 107.5) && (*vp > 144.5) && (*vp <= 149.5) && (*yp > 107.5) && (*yp <= 144.5) && (*up <= 100.5) ){
								if (draw){
									*yp = 255;  // make pixel brighter in image
								}
								kernel_cnt++;
							}
							if( (*up <= 107.5) && (*vp <= 144.5) && (*yp <= 101.5) && (*yp > 97.5) && (*vp <= 138.5) && (*up > 93.5) ){
								if (draw){
									*yp = 255;  // make pixel brighter in image
								}
								kernel_cnt++;
							}
							if( (*up > 107.5) && (*up <= 110.5) && (*vp > 141.5) && (*vp <= 144.5) && (*yp > 125.5) && (*yp <= 151.5) ){
								if (draw){
									*yp = 255;  // make pixel brighter in image
								}
								kernel_cnt++;
							}
            }
          } 
        }
      }

      // Draw the vectors in the vector list
      if (kernel_cnt > threshold){
        if (state == 0) {
          state = 1;
          kernel_centroid = kernel_size * (x_k + 1);
          vector_array[y_k] = kernel_centroid;
        }
      }
    }

    // If no vector for current y value is drawn yet, draw a vector with length 0
    if (state == 0) {
      state = 1;
      kernel_centroid = 0;
      // PRINT("Vector length %d\n", kernel_centroid);
      // PRINT("Yk value %d\n", y_k);
      vector_array[y_k] = kernel_centroid;
    }
  }

  // Draw the vectors on the screen


  





  if (draw){
    int16_t max = 0;
    int8_t vector_count = 0;
    for (int16_t y = half_kernel_size; y < (img->h - kernel_size); y+= kernel_size){
      max = vector_array[vector_count];
      if (max<0) {
        max = 0;
      }
      if (max > img->w) {
        max = img->w;
      }
      for(int16_t x = 0; x < max; x++){
        uint8_t *yp, *up, *vp;
        pix_values = compute_pixel_yuv(img, x, y);
        yp = pix_values.yp;
        up = pix_values.up;
        vp = pix_values.vp;
        *up = 0;
        *vp = 255;
        *yp = 125;
      }
      vector_count++;
    }
  }

  // Safety Triangle, adjust shape based on pitch
  float pitch  = DegOfRad((stateGetNedToBodyEulers_f()->theta)); //no float angle norm

  // T_x and T_y define height and width of triangle
  int16_t T_x = 4.0 * -1.0 * pitch;
  if (T_x < 0){
    T_x = 0;
  }
  if (T_x > 120){
    T_x = 120;
  }


  int16_t T_y = 190;
  float T_mid = vector_array_mid*kernel_size - half_kernel_size;

  // Calculate parameters for drawing triangle lines (x = alpha*y + beta)
  float alpha = T_x/(0.5 * T_y);
  float beta1 = T_x - alpha*(T_mid);
  float beta2 = T_x + alpha*(T_mid);

  
  bool in_triangle = true;

  // Loop through all vectors in vector array
  for (int8_t i = 0; i < vector_array_length; i++){
    int16_t vector_length = vector_array[i];

    int16_t y = i*kernel_size + half_kernel_size;

    // Test if vector is in triangle
    if (y > T_mid) {
        x = y*-1.0*alpha + beta2;
      }
    else {
      x = y*alpha + beta1;
    }

    // Find longest vector outside of safety triangle
    if (x <= 0){
      if (vector_length > vector_x){
        vector_x = vector_length;
        vector_y = y;
      }
    }
    else {
      if (vector_length < x) {
        in_triangle = false;
      }
    }

    if(!in_triangle) {
      cnt = 0;
    }
    else{
      cnt = vector_array[vector_array_mid];
    }
  }
  
  // Draw safety triangle and max length vector on screen
  if (draw){
    int16_t x = 0;

    for (int16_t y = T_mid - T_y/2; y < T_mid + T_y/2; y++){
      if (y > T_mid) {
        x = y*-1.0*alpha + beta2;
      }
      else {
        x = y*alpha + beta1;
      }

        uint8_t *yp, *up, *vp;
        pix_values = compute_pixel_yuv(img, x, y);
        yp = pix_values.yp;
        up = pix_values.up;
        vp = pix_values.vp;

        if (in_triangle){
          *up = 128;
          *vp = 0;
          *yp = 128;
        }
        else {
          *up = 128;
          *vp = 255;
          *yp = 128;
        }
      }

      int16_t max = vector_x;
      y = vector_y;
      if (max<0) {
        max = 0;
      }
      if (max > img->w) {
        max = img->w;
      }
      for(int16_t x = 0; x < max; x++){
        uint8_t *yp, *up, *vp;
        pix_values = compute_pixel_yuv(img, x, y);
        yp = pix_values.yp;
        up = pix_values.up;
        vp = pix_values.vp;
        *up = 192;
        *vp = 128;
        *yp = 20;
      }
    }
  /*
  Adviced direction predictor
  Use x = ay**2 + by + c for drawing parabols
  Draw parabolic around x axis and translate for plotting and vector comparison
  b = 0
  c is adjustable
  y_cross = 200
  a = -c/y_cross**2
  */
  int16_t c = 50;
  int16_t y_cross = 400;
  float a = -(float)c/(y_cross * y_cross);
  int16_t c_array[] = {25, 40, 50, 60};
  int16_t c_array_size = 4;

  int16_t x_ref = 0;
  int16_t y_ref = 0;

  int32_t direction = 0;
  int16_t direction_saturation = 50;

  // Construct direction predictor arrays
  for (int16_t i = c_array_size - 1; i >= 0; i--)
  {
    int16_t difference = 0;
    for (int16_t j = -1; j < 2; j += 2)
    {
      uint16_t temp = 0;
      for (int16_t n = 0; n < vector_array_mid; n++) {
        x = vector_array[vector_array_mid + n*j];
        y_ref = (n*kernel_size + half_kernel_size + 1)*j;
        x_ref = (int)(a*y_ref*y_ref) + c_array[i];

        if (draw) {
          // Draw pink vectors in middle, two layer thick for extra visibility
          uint8_t *yp, *up, *vp;
          pix_values = compute_pixel_yuv(img, x_ref, T_mid+n*j);
          yp = pix_values.yp;
          up = pix_values.up;
          vp = pix_values.vp;
          *up = 128;
          *vp = 255;
          *yp = 128;          
          pix_values = compute_pixel_yuv(img, x_ref+1, T_mid+n*j);
          yp = pix_values.yp;
          up = pix_values.up;
          vp = pix_values.vp;
          *up = 128;
          *vp = 255;
          *yp = 128;
        }

        // Stop if object is found
        temp = n;
        if (x < x_ref) {
          
          break;
        }
      }
      // Right away calculate the vector difference for predictive heading
      difference += j*temp;
    }

    // Do for every vector
    direction += difference;
  }
  // Bound vector advice such that a high spin situation is prevented
  Bound(direction, -direction_saturation, direction_saturation);



  // Draw the heading advice
  if (draw) {
    // Draw direction
    if (direction >= 0) {
      for (int16_t i = 0; i < direction; i++) {
        x = 20;
        y = T_mid + i;
        uint8_t *yp, *up, *vp;
        pix_values = compute_pixel_yuv(img, x, y);
        yp = pix_values.yp;
        up = pix_values.up;
        vp = pix_values.vp;
        *up = 230;
        *vp = 100;
        *yp = 29;
      }
    }
    else {
      for (int16_t i = 0; i > direction; i--) {
        x = 20;
        y = T_mid + i;
        uint8_t *yp, *up, *vp;
        pix_values = compute_pixel_yuv(img, x, y);
        yp = pix_values.yp;
        up = pix_values.up;
        vp = pix_values.vp;
        *up = 230;
        *vp = 100;
        *yp = 29;
      }
    }


    // Draw grey lines to indicate test area for predictive routing
    Bound(y_cross, 0, T_mid);
    int16_t x = 0;
    for (int16_t i = 0; i < c_array_size; i++) {
      c = c_array[i];
      for (int16_t y = T_mid - y_cross; y < T_mid + y_cross; y++){
        int16_t y_temp = y - T_mid;
          x = a*y_temp*y_temp + c;
          uint8_t *yp, *up, *vp;
          pix_values = compute_pixel_yuv(img, x, y);
          yp = pix_values.yp;
          up = pix_values.up;
          vp = pix_values.vp;
          *up = 128;
          *vp = 128;
          *yp = 128;
        }
    }
  }
  test.color_count = cnt;
  test.vector_x = vector_x;
  test.vector_y = vector_y;
  test.ptr = &mask_array;
  test.direction = direction;
  return test;
}

void color_object_detector_periodic(void)
{
  static struct color_object_t local_filters[1];
  pthread_mutex_lock(&mutex);
  memcpy(local_filters, global_filters, sizeof(struct color_object_t));
  pthread_mutex_unlock(&mutex);

  if(local_filters[0].updated){
    AbiSendMsgVISUAL_DETECTION(COLOR_OBJECT_DETECTION1_ID, local_filters[0].direction, local_filters[0].y_c,
        local_filters[0].vector_x, local_filters[0].vector_y, local_filters[0].color_count, 0);
    local_filters[0].updated = false;
  }
}
