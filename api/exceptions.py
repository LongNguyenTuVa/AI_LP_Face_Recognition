from flask import jsonify

class ErrorResponse(Exception):

    error_dict = {
        400 : '[image] field cannot be empty',
        401 : '[user_id] field cannot be empty',
        402 : 'invalid image file, only accept .png, .jpg or .jpeg file',
        403 : 'cannot read image',
        404 : 'cannot detect face from image',
        405 : 'user not found',
        406 : 'can not detect license plate from image',
        407 : 'cannot recognize license plate from image',
    }

    def __init__(self, error_code):
        Exception.__init__(self)
        self.message = self.error_dict.get(error_code, '')
        self.status_code = 200
        self.error_code = error_code

    def to_dict(self):
        response = dict(())
        response['result_code'] = self.error_code
        response['message'] = self.message
        return response