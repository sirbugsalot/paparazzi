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
#define in_nps 1
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
              }
          }

          else { // Filter for real life, decision tree implemented
            if( (*up <= 111.5) && (*vp <= 143.5) && (*yp > 93.5) && (*yp <= 160.5) ){
              if (draw){
                *yp = 255;  // make pixel brighter in image
              }
              kernel_cnt++;
            }       
            if( (*up > 111.5) && (*up <= 115.5) && (*vp <= 137.5) && (*yp > 96.5) ) {
              if (draw){
                *yp = 255;  // make pixel brighter in image
              }
              kernel_cnt++;
            }       
            if( (*up <= 111.5) && (*vp > 143.5) && (*vp <= 146.5) && (*yp > 108.5) ) {
              if (draw){
                *yp = 255;  // make pixel brighter in image
              }
              kernel_cnt++;
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

  int16_t vector_x = 0;
  int16_t vector_y = 0;
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
