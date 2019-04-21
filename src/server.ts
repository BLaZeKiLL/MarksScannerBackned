import express from 'express';
import path from 'path';
import multer from 'multer';

import { PythonShell, Options } from 'python-shell';

const app = express();

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views/'));

app.get('/', (req, res) => {
  res.render('test');
});

const py_options : Options = {
  mode: 'text',
  pythonPath: 'C:/ML_Notebook/env/Scripts/python.exe',
  pythonOptions: ['-u', '-W ignore'],
  scriptPath: __dirname + '/../python'
};

const storage = multer.diskStorage({
  destination: './uploads',
  filename: (req, file, cb) => cb(undefined, file.originalname)
});

app.post('/upload', multer({storage: storage, limits: { fieldSize: 5 * 1024 * 1024 }}).single('upload'), (req, res) => {
  // console.log(req.file.filename);

  py_options.args = [req.file.filename];

  PythonShell.run('api_final.py', py_options, (err, result) => {
    if (err) throw err;
    console.log(JSON.stringify(JSON.parse(result[0])));
    res.json({code: 200, message: 'OK', result: JSON.parse(result[0])});
  });

});

app.listen(3000, '10.167.1.178', () => {
  console.log('Serving at http://10.167.1.178:3000');
  console.log(__dirname);
});
