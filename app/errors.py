from flask import render_template, request, jsonify
from . import app


@app.errorhandler(403)
def forbidden(e):
    if not request.accept_mimetypes.accept_html and request.accept_mimetypes.accept_json:
        response = jsonify({'error': 'Forbidden'})
        response.status_code = 403
        return response
    return render_template('403.html'), 403


@app.errorhandler(404)
def page_not_found(e):
    if not request.accept_mimetypes.accept_html and request.accept_mimetypes.accept_json:
        response = jsonify({'error': 'Not found'})
        response.status_code = 404
        return response
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    if not request.accept_mimetypes.accept_html and request.accept_mimetypes.accept_json:
        response = jsonify({'error': 'Internal server error'})
        response.status_code = 500
        return response
    return render_template('500.html'), 500
