const url = "https://min-api.cryptocompare.com/data/pricemulti?fsyms=BTC,ETH&tsyms=USD,EUR";
const url2 = "http://127.0.0.1:3001/result"


let vm =new Vue({
    el: '#app',
    data: {
        array: null,
        first: null,
        second: null,
        third: null
    },

    created() {
        axios
        .get(url2, {params: {search: window.location.search.substring(1)}})
        .then(response => {
            this.array = response.data
            this.first = this.array.slice(0,5)
            this.second = this.array.slice(5,8)
            this.third = this.array.slice(8,10)
        })

    },
    
    methods: {
        sendmsg: function(){
            //alert(getQueryVariable('search'))
            axios
            .get(url2, {params: {search: window.location.search.substring(1)}})
            .then(response => {
                this.array = response.data
            })
        }
    },
})

Vue.component('news-test', {
    props : ['result'],
    template :  `
        <div>
        <dt> {{result.type}} </dt>
        <dd> {{result.percent}} </dd>
        </div>
    `
})