var log_file = "/etc/nginx/logs/log_http.txt";

function hello(r) {
    r.return(200, "Hello world!");
}

async function logsProcess(s){
    var lines     
    lines = require('fs').readFileSync(log_file, 'utf-8').split('\n').filter(Boolean);
    
    try{
        lines.forEach(async (line) => {
            var jdata = JSON.parse(line.toString())
            await ngx.fetch('http://app:5000/log/api/requests',{method:'POST', body:JSON.stringify(jdata)})
        })
    
        require('fs').writeFileSync(log_file, '', 'utf-8');
    } catch(e) {
        ngx.log(ngx.ERR, "Falhou");
    }
}

export default {hello, logsProcess};