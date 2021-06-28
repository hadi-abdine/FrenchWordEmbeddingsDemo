 
$(document).ready(function() {
    function copyToClipboard(text) {
        var dummy = document.createElement("textarea");
        document.body.appendChild(dummy);
        //Be careful if you use texarea. setAttribute('value', value), which works with "input" does not work with "textarea". – Eduard
        dummy.value = text;
        dummy.select();
        document.execCommand("copy");
        document.body.removeChild(dummy);
    }

    $('.rating').hide();
    $('.ratingTitle').hide();

    $("#abstract_output").bind("contextmenu",function(e){
        return false;
    });

    $('#abstract_output').keypress(function(e) {
        e.preventDefault();
    });

    $("#title_output").bind("contextmenu",function(e){
        return false;
    });

    $('#title_output').keypress(function(e) {
        e.preventDefault();
    });

    $('#star5').on('click', function(e) {
        e.preventDefault()
        var text_input = $('#text_input1').val();
        var test = $('#abstract_output').val();
        var label = 5;
        copyToClipboard(test)

        res = $.ajax({
            url : '/FrenchLinguisticResources/barthezSumRating',
            type : 'POST',
            data : JSON.stringify({ fullText : text_input, summary : test, label : label}),
            dataType: "json",
            contentType: "application/json",
        });


        $('.rating').fadeOut("slow");

    });

    $('#star4').on('click', function(e) {
        e.preventDefault()
        var text_input = $('#text_input1').val();
        var test = $('#abstract_output').val();
        var label = 4;
        copyToClipboard(test)

        res = $.ajax({
            url : '/FrenchLinguisticResources/barthezSumRating',
            type : 'POST',
            data : JSON.stringify({ fullText : text_input, summary : test, label : label}),
            dataType: "json",
            contentType: "application/json",
        });


        $('.rating').fadeOut("slow");

    });

    $('#star3').on('click', function(e) {
        e.preventDefault()
        var text_input = $('#text_input1').val();
        var test = $('#abstract_output').val();
        var label = 3;
        copyToClipboard(test)

        res = $.ajax({
            url : '/FrenchLinguisticResources/barthezSumRating',
            type : 'POST',
            data : JSON.stringify({ fullText : text_input, summary : test, label : label}),
            dataType: "json",
            contentType: "application/json",
        });


        $('.rating').fadeOut("slow");

    });

    $('#star2').on('click', function(e) {
        e.preventDefault()
        var text_input = $('#text_input1').val();
        var test = $('#abstract_output').val();
        var label = 2;
        copyToClipboard(test)

        res = $.ajax({
            url : '/FrenchLinguisticResources/barthezSumRating',
            type : 'POST',
            data : JSON.stringify({ fullText : text_input, summary : test, label : label}),
            dataType: "json",
            contentType: "application/json",
        });


        $('.rating').fadeOut("slow");

    });

    $('#star1').on('click', function(e) {
        e.preventDefault()
        var text_input = $('#text_input1').val();
        var test = $('#abstract_output').val();
        var label = 1;
        copyToClipboard(test)

        res = $.ajax({
            url : '/FrenchLinguisticResources/barthezSumRating',
            type : 'POST',
            data : JSON.stringify({ fullText : text_input, summary : test, label : label}),
            dataType: "json",
            contentType: "application/json",
        });


        $('.rating').fadeOut("slow");

    });

    $('#star5t').on('click', function(e) {
        e.preventDefault()
        var text_input = $('#text_input2').val();
        var test = $('#title_output').val();
        var label = 5;
        copyToClipboard(test)

        res = $.ajax({
            url : '/FrenchLinguisticResources/barthezTitleRating',
            type : 'POST',
            data : JSON.stringify({ fullText : text_input, summary : test, label : label}),
            dataType: "json",
            contentType: "application/json",
        });


        $('.ratingTitle').fadeOut("slow");

    });

    $('#star4t').on('click', function(e) {
        e.preventDefault()
        var text_input = $('#text_input2').val();
        var test = $('#title_output').val();
        var label = 4;
        copyToClipboard(test)

        res = $.ajax({
            url : '/FrenchLinguisticResources/barthezTitleRating',
            type : 'POST',
            data : JSON.stringify({ fullText : text_input, summary : test, label : label}),
            dataType: "json",
            contentType: "application/json",
        });


        $('.ratingTitle').fadeOut("slow");

    });

    $('#star3t').on('click', function(e) {
        e.preventDefault()
        var text_input = $('#text_input2').val();
        var test = $('#title_output').val();
        var label = 3;
        copyToClipboard(test)

        res = $.ajax({
            url : '/FrenchLinguisticResources/barthezTitleRating',
            type : 'POST',
            data : JSON.stringify({ fullText : text_input, summary : test, label : label}),
            dataType: "json",
            contentType: "application/json",
        });


        $('.ratingTitle').fadeOut("slow");

    });

    $('#star2t').on('click', function(e) {
        e.preventDefault()
        var text_input = $('#text_input2').val();
        var test = $('#title_output').val();
        var label = 2;
        copyToClipboard(test)

        res = $.ajax({
            url : '/FrenchLinguisticResources/barthezTitleRating',
            type : 'POST',
            data : JSON.stringify({ fullText : text_input, summary : test, label : label}),
            dataType: "json",
            contentType: "application/json",
        });


        $('.ratingTitle').fadeOut("slow");

    });

    $('#star1t').on('click', function(e) {
        e.preventDefault()
        var text_input = $('#text_input2').val();
        var test = $('#title_output').val();
        var label = 1;
        copyToClipboard(test)

        res = $.ajax({
            url : '/FrenchLinguisticResources/barthezTitleRating',
            type : 'POST',
            data : JSON.stringify({ fullText : text_input, summary : test, label : label}),
            dataType: "json",
            contentType: "application/json",
        });


        $('.ratingTitle').fadeOut("slow");

    });


    $('.get_summary').on('click', function(e) {

        $('#abstract_output').bind("cut copy paste",function(e) {
            e.preventDefault();
        });
    
        $("#abstract_output").bind("contextmenu",function(e){
            return false;
        });

        
        $("#error").fadeIn("slow");
        setTimeout(function(){
        $("#error").fadeOut("slow");
        },2000);

        var text_input = $('#text_input1').val();
        var abstract = $('#abstract_output').val();

        req = $.ajax({
            url : '/FrenchLinguisticResources/barthezSum',
            type : 'POST',
            data : JSON.stringify({ fullText : text_input, abstract : abstract}),
            dataType: "json",
            contentType: "application/json",
        });

        req.done(function(data) {
            $('#abstract_output').val(data.abstract);
            $('.rating').fadeIn("slow");

        });
    });

    
    $('.get_title').on('click', function() {

        var text_input = $('#text_input2').val();
        var title = $('#title_output').val();

        req = $.ajax({
            url : '/FrenchLinguisticResources/barthezTitle',
            type : 'POST',
            data : JSON.stringify({fullText : text_input, title : title}),
            error: function(e) {
                console.log(e);
            },
            dataType: "json",
            contentType: "application/json",
        });

        req.done(function(data) {
            $('#title_output').val(data.title);
            $('.ratingTitle').fadeIn("slow");
        });
    });

    $('.get_scores').on('click', function() {

        var text_input = $('#text_input3').val();
        var n = "25%"
        var p = "75%"
        req = $.ajax({
            url : '/FrenchLinguisticResources/barthezSentiment',
            type : 'POST',
            data : JSON.stringify({fullText : text_input, p : p, n :  n}),
            error: function(e) {
                console.log(e);
            },
            dataType: "json",
            contentType: "application/json",
        });

        req.done(function(data) {
            $(".posi").animate({
                width: data.p
              }, 2500);
            $(".negi").animate({
                width: data.n
                }, 2500);
            $("#poslabel").text("Positive".concat(" : ", data.p));
            $("#neglabel").text("Negative".concat(" : ", data.n));
        });
    });

    $('.get_words').on('click', function() {

        var text_input = $('#text_input4').val();

        req = $.ajax({
            url : '/FrenchLinguisticResources/barthezMLM',
            type : 'POST',
            data : JSON.stringify({fullText : text_input}),
            error: function(e) {
                console.log(e);
            },
            dataType: "json",
            contentType: "application/json",
        });

        req.done(function(data) {

            $("#l1").text(data.w1.concat(" : ", data.p1));
            $("#l2").text(data.w2.concat(" : ", data.p2));
            $("#l3").text(data.w3.concat(" : ", data.p3));
            $("#l4").text(data.w4.concat(" : ", data.p4));
            $("#l5").text(data.w5.concat(" : ", data.p5));
            $(".w1").animate({
                width: data.s1
              }, 2500);
            $(".w2").animate({
                width: data.s2
                }, 2500);
            $(".w3").animate({
                width: data.s3
                }, 2500);
            $(".w4").animate({
                width: data.s4
                }, 2500);
            $(".w6").animate({
                width: data.s5
                }, 2500);

                            
        });
    });



    $('.get_off_label').on('click', function() {

        var text_input = $('#text_input5').val();
        req = $.ajax({
            url : '/FrenchLinguisticResources/bertweetOff',
            type : 'POST',
            data : JSON.stringify({fullText : text_input}),
            error: function(e) {
                console.log(e);
            },
            dataType: "json",
            contentType: "application/json",
        });

        req.done(function(data) {
            $(".posi2").animate({
                width: data.p
              }, 2500);
            $(".negi2").animate({
                width: data.n
                }, 2500);
            $("#poslabel2").text("Not Offensive".concat(" : ", data.p));
            $("#neglabel2").text("Offensive".concat(" : ", data.n));
        });
    });

     $('.get_ner').on('click', function(e) {

        var text_input = $('#text_input6').val();

        req = $.ajax({
            url : '/FrenchLinguisticResources/bertweetNER',
            type : 'POST',
            data : JSON.stringify({ fullText : text_input}),
            dataType: "json",
            contentType: "application/json",
        });

        req.done(function(data) {
            $('#nerres').val(data.res);
        });
    });
});





