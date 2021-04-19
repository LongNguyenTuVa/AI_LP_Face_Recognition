## Facial & License Plate Recognition API Docs
### 1. Recognize a License Plate
Recognize license plate from input image, return the text on the plate, the plate image and the confidents
**URL** : `/api/license_plate/recognize`
**Method** : `POST` (multipart/form-data)
**Body** :
> *key* : `image` (not null & only accept PNG, JPG, JPEG)
#### **Success Response**
**Code** : `200 OK`
**Content example**
```json
{
    "detection_conf": "100%",
    "image_path": "static/images/license_plates/20210419_105659_PIH1Q6_lp.jpg",
    "recognition_conf": "47%",
    "text": "30M-2943"
}
```
#### Error Response
**Condition** : If `image` key is null
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "[image] field can not empty"
}
```
**Condition** : If can not detect liense plate from input image
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "can not detect license plate from image"
}
```
### 2. Register a new Driver
Recognize a new driver or update the image for the existing driver
**URL** : `api/face/register`
**Method** : `POST` (multipart/form-data)
**Body** :
> user_id : `int` optional (If user_id is null, new user will be registered. If user_id is not null, the current user with this id will be updated)
> key : `images` (array of image, not null & only accept PNG, JPG, JPEG). In case register new user, this is required at least 3 images
#### **Success Response**
**Code** : `200 OK`
**Content example**
```json
{
    "user_id": "2"
}
```
#### Error Response
**Condition** : If `images` key is null
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "[images] field can not empty"
}
```
**Condition** : If register a new user with less than 3 images
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "register new face with at least 3 images"
}
```
```
**Condition** : If there is a problem with detect face from some images
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "registering new user requires at least 3 images containing user's face. Only 2 images contain the user's face, please re-register image image1.jpg with another image."
}
```
### 3. Recognize a Driver
Recognize drive from input image, return the driver_id, the driver face image and the confidents
**URL** : `api/face/recognize`
**Method** : `POST` (multipart/form-data)
**Body** :
> *key* : `image` (not null & only accept PNG, JPG, JPEG)
#### **Success Response**
**Code** : `200 OK`
**Content example**
```json
{
    "detection_conf": "100%",
    "image_path": "static/images/faces/20210419_120309_OSEJS4_face.jpg",
    "recognition_distance": "1.00",
    "user_id": 1
}
```
#### Error Response
**Condition** : If can not recognize the driver
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "user not found"
}
```
**Condition** : If `image` key is null
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "[image] field can not empty"
}
```
**Condition** : If can not detect face from input image
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "can not detect face from image"
}
```