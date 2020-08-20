 
$(document).ready(function() {

    $('.analogyBtn').on('click', function() {

        var word1 = $('#word1').val();
        var word2 = $('#word2').val();
        var word3 = $('#word3').val();
        var word4 = $('#word4').val();

        req = $.ajax({
            url : '/analogy',
            type : 'POST',
            data : { word1 : word1, word2 : word2, word3 : word3, word4 : word4}
        });

        req.done(function(data) {

            $('#word4').val(data.word_4);

        });
    

    });


    $('.simScoreBtn').on('click', function() {

        var sim1 = $('#sim1').val();
        var sim2 = $('#sim2').val();
        var simscore = $('#simscore').val();

        req = $.ajax({
            url : '/similarityscore',
            type : 'POST',
            data : { sim1 : sim1, sim2 : sim2, simscore : simscore}
        });

        req.done(function(data) {

            $('#simscore').val(data.simscore);

        });
    

    });


    $('.simWordsBtn').on('click', function() {

        var wordgoal = $('#wordgoal').val();
        var simwords = $('#simwords').val();

        req = $.ajax({
            url : '/similaritywords',
            type : 'POST',
            data : { wordgoal : wordgoal, simwords : simwords}
        });

        req.done(function(data) {

            $('#simwords').val(data.simwords);

        });
    

    });

});