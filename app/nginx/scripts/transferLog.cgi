#!/bin/bash

input="/etc/nginx/logs/log_http.txt"
declare -i lineno=0
while IFS= read -r line
do
    response=$(curl -s -w '%{http_code}' -H 'Content-Type: application/json' -d "$(echo $line | jq '.')" -X POST http://localhost:8081/api/requests)
    
    http_code=$(tail -n1 <<< "$response") 

    echo $http_code

    if [ $http_code -eq "200" ]
    then
        let ++lineno
        sed -i "1 d" "$input"
    fi
done < "$input" 