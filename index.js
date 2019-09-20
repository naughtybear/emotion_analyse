var express = require('express')
var app = express()
const spawn = require('child_process').spawn;
const model = spawn('/home/naughtybear/anaconda3/envs/py36/bin/python', ['test_model.py'])
in_num = 1

app.use('/view', express.static('view'))

app.get('/', (req, res)=>{
    res.sendFile(__dirname + '/index.html')
})

app.get('/result', (req, res)=>{
    //res.write('<head><meta charset="utf-8"/></head>');
    res.header('Access-Control-Allow-Origin', '*');
    console.log(req.query.search);
    count = 0
    model.stdout.on('data', function(data){
        count++
        if(count == in_num){
            in_num++
            console.log(data);
            
            let array = []
            data = String(data).split('\n')
            for(let i=0; i<data.length; i++){
                //res.write('<div>' + data[i] + '</div>')
                let buf = data[i].split('\n')
                if(i==0 || i==6){
                    var tmp = {
                        "id" : i,
                        "type" : buf[0],
                        "percent" : " "
                    }
                }
                else{
                    var tmp = {
                        "id" : i,
                        "type" : buf[0],
                        "percent" : buf[1]
                    }
                }
                array.push(tmp)
            }      
            res.send(array)      
            res.end()
        }
        //res.write('<div>' + data + '</div>')
    })
    //console.log("before write")
    model.stdin.write(req.query.search+'\n')
    //console.log("after write")
})

app.get('/show', function (req, res){
    res.sendFile(__dirname + '/test.html')
})

var server = app.listen(3001, function(){
    console.log ('Server跑起來了，現在時間是:' + new Date())
    console.log(__dirname)
})