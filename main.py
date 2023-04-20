from flask import Flask, render_template, request, redirect, url_for,session
import cv2
import numpy as np
import os
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

# from script import grade_workpiece

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "DOMATORRETO"
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the template image from the request
        template_image = request.files['template_image']
        
        # Save the template image to a temporary directory on the server
        template_path = os.path.join(app.config['UPLOAD_FOLDER'], template_image.filename)
        template_image.save(template_path)
        
        # Store the path to the template image in the session
        session['template_path'] = template_path
        
        # Redirect the user to the upload page
        return redirect(url_for('upload'))
    
    # Render the home page with the template image upload form
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    template_path = session.get('template_path', None)
    if request.method == 'POST':
        print("Post request from upload initiated")
        file = request.files['file']
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #get the template
        # template = cv2.imread('/home/satarw/Documents/mini_proj/prototype/test_imgs/template.jpg', cv2.IMREAD_GRAYSCALE)
        template = cv2.imread(template_path,cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]

        #template matching using opencv2
        res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        img = img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]] #this crops out the area where the template was found

        #resize image to reduce computational load  
        base_size = 256
        h1,h2 = int(base_size*img.shape[0]/img.shape[1]), int(base_size*template.shape[0]/template.shape[1])
        img = cv2.resize(img,(base_size,h1))
        template = cv2.resize(template,(base_size,h2))

        #flatten image to find pearson correlation coefficient
        img_flat = img.flatten()
        template_flat = template.flatten()
        pc = round(np.corrcoef(img_flat,template_flat)[0][1],3)
        # print("The correlation with template is ",pc)
        result = ""
        if pc >= 0.9:
            result = "Grade for this workpiece = {}".format(10)
        elif pc >= 0.86:
            result = "Grade for this workpiece = {}".format(9)
        elif pc >=0.7:
            result = "Grade for this workpiece = {}".format(8)
        else:
            result = "Incorrect image, please check the image and positioning of the workpiece"

        # img_str = base64.b64encode(cv2.imencode('.png', img)[1]).decode() 
        return redirect(url_for('results', result=result))
    return render_template('upload.html')

@app.route('/results', methods = ['GET','POST'])
def results():
    # img_data = request.args.get('img_path')
    result = request.args.get('result')
    # img_data = base64.b64decode(img_data)
    # img = cv2.imdecode(np.fromstring(img_data, np.uint8), cv2.IMREAD_COLOR)
    return render_template('results.html',result=result)

if __name__ == '__main__':
    app.run(debug=True)
