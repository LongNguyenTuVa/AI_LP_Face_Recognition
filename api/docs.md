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
    "detection_conf": "86",
    "image_path": "static/images/license_plates/20210426_124125_IHN88Q_lp.jpg",
    "recognition_conf": "97",
    "text": "30V-7996"
}
```
#### Error Response
**Condition** : If `image` key is null
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "[image] field cannot be empty"
}
```
**Condition** : If can not detect liense plate from input image
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "cannot detect license plate from image"
}
```
### 2. Register a new Driver
Recognize a new driver or update the image for the existing driver
**URL** : `http://sso.d2s.com.vn:5000/api/face/register`
**Method** : `POST` (multipart/form-data)
**Body** :
> *user_id* : `string` (not null)
> *key* : `image` (not null & only accept PNG, JPG, JPEG)
#### **Success Response**
**Code** : `200 OK`
**Content example**
```json
{
    "message": "successfully registered"
}
```
#### Error Response
**Condition** : If `user_id` or `image` key is null
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "[user_id] field cannot be empty"
}
```
```json
{
    "error": "[image] field cannot be empty"
}
```
**Condition** : If can not detect face from input image
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "cannot detect face from image"
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
    "detection_conf": "100",
    "image_path": "static/images/faces/20210426_130000_4RO4PM_face.jpg",
    "recognition_distance": "95",
    "user_id": "hoa_vu"
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
    "error": "[image] field cannot be empty"
}
```
**Condition** : If can not detect face from input image
**Code** : `400 BAD REQUEST`
**Content** :
```json
{
    "error": "cannot detect face from image"
}
```