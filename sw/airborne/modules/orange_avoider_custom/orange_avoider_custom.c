/*
 * Copyright (C) Roland Meertens
 *
 * This file is part of paparazzi
 *
 */
/**
 * @file "modules/orange_avoider/orange_avoider.c"
 * @author Roland Meertens
 * Example on how to use the colours detected to avoid orange pole in the cyberzoo
 * This module is an example module for the course AE4317 Autonomous Flight of Micro Air Vehicles at the TU Delft.
 * This module is used in combination with a color filter (cv_detect_color_object) and the navigation mode of the autopilot.
 * The avoidance strategy is to simply count the total number of orange pixels. When above a certain percentage threshold,
 * (given by color_count_frac) we assume that there is an obstacle and we turn.
 *
 * The color filter settings are set using the cv_detect_color_object. This module can run multiple filters simultaneously
 * so you have to define which filter to use with the ORANGE_AVOIDER_VISUAL_DETECTION_ID setting.
 */

#include "modules/orange_avoider_custom/orange_avoider_custom.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <math.h>

#define NAV_C // needed to get the nav functions like Inside...
#include "generated/flight_plan.h" //do not change this

#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[orange_avoider_custom->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif

static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
//static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t change_nav_heading(float heading_change, float heading_increment);
static uint8_t defineNewHeading(void);

//static uint8_t chooseRandomIncrementAvoidance(void);

enum navigation_state_t {
  SAFE,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS
};

// define and initialise global variables
enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
int32_t confidence_value = 0;   // 0 = no obstacle, 1 = obstacle         
int16_t obstacle_free_confidence = 0;   // a measure of how certain we are that the way ahead is safe.
int16_t turn_counter = 0;   //measure number of turns in a row
float heading_increment = 0.1f;          // heading angle increment [deg]
float heading_change = 10.f;          // heading angle change [deg]
float maxDistance = 2.25;               // max waypoint displacement [m]

int16_t pixelX = 120; //initial values
int16_t pixelY = 260;

bool new_message = false; //boolean to keep track of vision

// define settings
float oa_color_count_frac = 0.18f;  //if i delete this the autopilot.c file crashes

const int16_t max_trajectory_confidence = 6; // number of consecutive negative object detections to be sure we are obstacle free

/*
 * This next section defines an ABI messaging event (http://wiki.paparazziuav.org/wiki/ABI), necessary
 * any time data calculated in another module needs to be accessed. Including the file where this external
 * data is defined is not enough, since modules are executed parallel to each other, at different frequencies,
 * in different threads. The ABI event is triggered every time new data is sent out, and as such the function
 * defined in this file does not need to be explicitly called, only bound in the init function
 */
#ifndef ORANGE_AVOIDER_VISUAL_DETECTION_ID
#define ORANGE_AVOIDER_VISUAL_DETECTION_ID ABI_BROADCAST
#endif
static abi_event color_detection_ev;
static void color_detection_cb(uint8_t __attribute__((unused)) sender_id,
                               int16_t __attribute__((unused)) pixel_x, int16_t __attribute__((unused)) pixel_y,
                               int16_t pixel_width, int16_t pixel_height,
                               int32_t quality, int16_t __attribute__((unused)) extra)
{
  confidence_value = quality;  //proportional to length of middle vectors 0-240
  pixelX = pixel_width;  //x coordinates of optimal value
  pixelY = pixel_height; // y

  if (quality = true){
    //PRINT("VISION TRUE");
    new_message = true;  //message delivered set to true
  }
  
  // PRINT("COLOR COUNT IN ORANGE AVOIDER = %d", color_count);
  // PRINT("VX, VY in orange avoider = %d %d", vx, vy);
}

/*
 * Initialisation function, setting the colour filter, random seed and heading_increment
 */
void orange_avoider_init(void)
{
  // Initialise random values
  srand(time(NULL));
  //chooseRandomIncrementAvoidance();

  // bind our colorfilter callbacks to receive the color filter outputs
  AbiBindMsgVISUAL_DETECTION(ORANGE_AVOIDER_VISUAL_DETECTION_ID, &color_detection_ev, color_detection_cb);
}

/*
 * Function that checks it is safe to move forwards, and then moves a waypoint forward or changes the heading
 */
void orange_avoider_periodic(void)
{
  // only evaluate our state machine if we are flying
  if(!autopilot_in_flight()){
    return;
  }

  // update our safe confidence using confidence value (from vision)
  if(confidence_value > 30){ // there is no obstacle
    obstacle_free_confidence++;
    //VERBOSE_PRINT("OBS CONF + 1\n");
  } else {  // there is obstacle
    obstacle_free_confidence -= 2;  // be more cautious with positive obstacle detections
    //VERBOSE_PRINT("OBS CONF - 2\n");
  }

  // bound obstacle_free_confidence
  Bound(obstacle_free_confidence, 0, max_trajectory_confidence);

  float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence); //0.2

  switch (navigation_state){
    case SAFE:
      // Move waypoint forward
      VERBOSE_PRINT(" -- SAFE: conf_val %d , obstac_conf %d, (x,y) = %d, %d\n", confidence_value, obstacle_free_confidence, pixelX, pixelY);
      moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);
      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
        navigation_state = OUT_OF_BOUNDS;
      } else if (obstacle_free_confidence == 0){
        navigation_state = OBSTACLE_FOUND;
      } else {
        moveWaypointForward(WP_GOAL, 1.2f *moveDistance);
      }

      break;
    case OBSTACLE_FOUND: // logic: stop and define heading change angle and heading increment based on x/y pixel
      VERBOSE_PRINT(" -- OBSTACLE FOUND\n");
      // stop
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_TRAJECTORY);

      // define 'heading_change' and 'heading_increment'based on either optimal path found by vision or a set large angle
      defineNewHeading();

      navigation_state = SEARCH_FOR_SAFE_HEADING;

      break;
    case SEARCH_FOR_SAFE_HEADING: // logic: turn by defined heading change with defined heading increment. Then check if safe to proceed
      VERBOSE_PRINT(" -- SEARCH HEADING: conf_val %d , obstac_conf %d, (x,y) = %d, %d\n", confidence_value, obstacle_free_confidence, pixelX, pixelY);
      // turn by 'heading change' in steps of 'heading increment'
      
      if ((new_message) && (obstacle_free_confidence == 0)){  //wait for vision
          //VERBOSE_PRINT(" -- CHANGING HEADING---"); 
          change_nav_heading(heading_change, heading_increment);
          //turn_counter++; // within function
          VERBOSE_PRINT(" -- VISION TRUE - change heading, turn counter: %d\n", turn_counter);
      }
      new_message = false; //force waiting for new vision input
      //VERBOSE_PRINT(" -- VISION FALSE ---\n"); 

      // After turning check if heading is free to continue (with certain confidence)
        if (obstacle_free_confidence > 0){ //need to check thresholds cause this might run the turning function twice
          VERBOSE_PRINT(" Safe to continue ---\n"); 
          turn_counter == 0;
          navigation_state = SAFE;
        }
        //if (turn_counter > 3){
        //  VERBOSE_PRINT("I'M STUCK (90deg turn right)\n"); 
        //  change_nav_heading(100.f, 0.1f);
        //  turn_counter == 0;
        //}
      break;
    case OUT_OF_BOUNDS:
      VERBOSE_PRINT(" -- OUT OF BOUNCE\n");
      //defineNewHeading();
      change_nav_heading(100.f, 0.1f); //if our of bounce turn 90deg right
      moveWaypointForward(WP_TRAJECTORY, 1.5f);

      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
        // add offset to head back into arena
        //change_nav_heading(heading_change, heading_increment); //delete this?

        // reset safe counter
        obstacle_free_confidence = 0;

        // ensure direction is safe before continuing
        navigation_state = SEARCH_FOR_SAFE_HEADING;
      }
      break;
    default:
      break;
  }
  return;
}

/*
 * Increases the NAV heading. Assumes heading is an INT32_ANGLE. It is bound in this function.
 */
// uint8_t increase_nav_heading(float incrementDegrees)
// {
//   float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);

//   // normalize heading to [-pi, pi]
//   FLOAT_ANGLE_NORMALIZE(new_heading);

//   // set heading, declared in firmwares/rotorcraft/navigation.h
//   // for performance reasons the navigation variables are stored and processed in Binary Fixed-Point format
//   nav_heading = ANGLE_BFP_OF_REAL(new_heading);

//   VERBOSE_PRINT("Increasing heading to %f\n", DegOfRad(new_heading));
//   return false;
// }

/*
 * Calculates coordinates of distance forward and sets waypoint 'waypoint' to those coordinates
 */
uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return false;
}

/*
 * Change the objective waypoint 'goal' to next waypoint (A, B, C)
//  */
// uint8_t moveWaypointNext(uint8_t goal, uint8_t waypoint)
// {
//   struct EnuCoor_i wp_coor;
//   wp_coor->x = WaypointX(waypoint);
//   wp_coor->y = WaypointY(waypoint);
//   moveWaypoint(goal, &wp_coor);
//   return false;
// }

/*
 * Calculates coordinates of a distance of 'distanceMeters' forward w.r.t. current position and heading
 */
uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading  = stateGetNedToBodyEulers_f()->psi;

  // Now determine where to place the waypoint you want to go to
  new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * (distanceMeters));
  new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * (distanceMeters));
  //VERBOSE_PRINT("Calculated %f m forward position. x: %f  y: %f based on pos(%f, %f) and heading(%f)\n", distanceMeters,	
   //             POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y),
   //             stateGetPositionEnu_f()->x, stateGetPositionEnu_f()->y, DegOfRad(heading));
  return false;
}

/*
 * Sets waypoint 'waypoint' to the coordinates of 'new_coor'
 */
uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  //VERBOSE_PRINT("Moving waypoint %d to x:%f y:%f\n", waypoint, POS_FLOAT_OF_BFP(new_coor->x),
  //              POS_FLOAT_OF_BFP(new_coor->y));
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}

/*
 * Sets the variable 'heading_increment' randomly positive/negative
 */
// uint8_t chooseRandomIncrementAvoidance(void)
// {
//   // Randomly choose CW or CCW avoiding direction
//   if (rand() % 2 == 0) {
//     heading_increment = 5.f;
//     VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
//   } else {
//     heading_increment = -5.f;
//     VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
//   }
//   return false;
// }

/*
 * Sets the variable 'heading_change' based on vision information (set to either 90deg or an optimal heading)
 */
uint8_t defineNewHeading(void)
{

  VERBOSE_PRINT("Best Pixel (X,Y): (%d, %d)\n", pixelX, pixelY);

  // Uses x/y of optimal path/pixel to compute newheading
  if (pixelX < 30) {   // if horizon too low -> turn 90deg
    heading_change = 65.f;
    //VERBOSE_PRINT("Low Horizon (X<30) | Set heading_change to: %f deg\n",  heading_change);
  }  else{   // if horizon not too low -> turn based on optimal path
    heading_change = DegOfRad(atan(abs(260 - pixelY)/abs(pixelX))); //atan gives angle in radiants, so transform to degrees
    heading_change = fmaxf(heading_change, 10.); //choose between angle computed and 5deg, so min change is 5deg
    //VERBOSE_PRINT("Optimal path | Set heading_change to: %f deg\n",  heading_change);
  }

  //Define direction of turn based on y coord of pixel
  if ((260-pixelY) > 0) { // if pixel to the left of drone, turn ccw
    heading_increment = -0.1f;
    //VERBOSE_PRINT("Turn left (ccw): Y<260");
  }else{  //if pixel to right, turn cw
    heading_increment = 0.1f;
    //VERBOSE_PRINT("Turn right (cw): Y>260");
  }
  return false;
}

/*
 * Changes the NAV heading. Assumes heading is an INT32_ANGLE. 
 */

uint8_t change_nav_heading(float heading_change, float heading_increment)
{

  // new heading is current heading plus increments (until total change angle achieve)
  float total_turn = 0.; // variable to keep track of turn
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(heading_increment); //in rad   (defined outside while loop)
  //VERBOSE_PRINT("Heading before while loop is: %f deg\n", DegOfRad(stateGetNedToBodyEulers_f()->psi));
  while (total_turn < heading_change) {
    //VERBOSE_PRINT("TURNING, total turn is: %f out of %f\n", total_turn, heading_change);
    // normalize heading to [-pi, pi]
    FLOAT_ANGLE_NORMALIZE(new_heading); 

    //VERBOSE_PRINT("Normalised new heading is: %f", new_heading);
    // set heading, declared in firmwares/rotorcraft/navigation.h
    // for performance reasons the navigation variables are stored and processed in Binary Fixed-Point format
    nav_heading = ANGLE_BFP_OF_REAL(new_heading);

    total_turn += fabs(heading_increment);  //in deg
    new_heading += + RadOfDeg(heading_increment); //add to heading
  }
  //VERBOSE_PRINT("Heading after while loop is: %f deg\n", DegOfRad(stateGetNedToBodyEulers_f()->psi)); //this is no different than before loop...
  //if (total_turn >= heading_change){
   // turn_counter++;
   // VERBOSE_PRINT("TURN + 1");
  //}

  VERBOSE_PRINT("END TUNING - Increased heading by %f deg in steps of %f to %f\n deg", heading_change, heading_increment, DegOfRad(new_heading));
  return false;
}

