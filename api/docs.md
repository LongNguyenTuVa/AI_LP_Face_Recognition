## Facial & License Plate Recognition API Docs
### 1. Recognize a License Plate
Recognize license plate from input image, return the text on the plate, the plate image and the confidents
**URL** : `http://sso.d2s.com.vn:5000/api/license_plate/recognize`
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
**URL** : `http://sso.d2s.com.vn:5000/api/face/register`
**Method** : `POST` (multipart/form-data)
**Body** :
> user_id : `int` optional (If user_id is null, new user will be registered. If user_id is not null, the current user with this id will be updated)
> *key* : `image` (not null & only accept PNG, JPG, JPEG)
#### **Success Response**
**Code** : `200 OK`
**Content example**
```json
{
    "user_id": "2"
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
**Condition** : If can not detect face from input image
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "can not detect face from image"
}
```
### 3. Recognize a Driver
Recognize drive from input image, return the driver_id, the driver face image and the confidents
**URL** : `http://sso.d2s.com.vn:5000/api/face/recognize`
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