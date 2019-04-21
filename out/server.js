"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const path_1 = __importDefault(require("path"));
const multer_1 = __importDefault(require("multer"));
const python_shell_1 = require("python-shell");
const app = express_1.default();
app.set('view engine', 'ejs');
app.set('views', path_1.default.join(__dirname, 'views/'));
app.get('/', (req, res) => {
    res.render('test');
});
const py_options = {
    mode: 'text',
    pythonPath: 'C:/ML_Notebook/env/Scripts/python.exe',
    pythonOptions: ['-u', '-W ignore'],
    scriptPath: __dirname + '/../python'
};
const storage = multer_1.default.diskStorage({
    destination: './uploads',
    filename: (req, file, cb) => cb(undefined, file.originalname)
});
app.post('/upload', multer_1.default({ storage: storage, limits: { fieldSize: 5 * 1024 * 1024 } }).single('upload'), (req, res) => {
    // console.log(req.file.filename);
    py_options.args = [req.file.filename];
    python_shell_1.PythonShell.run('api_final.py', py_options, (err, result) => {
        if (err)
            throw err;
        console.log(JSON.stringify(JSON.parse(result[0])));
        res.json({ code: 200, message: 'OK', result: JSON.parse(result[0]) });
    });
});
app.listen(3000, '10.167.1.178', () => {
    console.log('Serving at http://10.167.1.178:3000');
    console.log(__dirname);
});
//# sourceMappingURL=server.js.map