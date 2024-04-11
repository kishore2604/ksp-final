/*
Template: LAWYER - Multipurpose Bootstrap 5 Template
Author: RRDevs
*/

(function($) {
    "use strict";

    $(document).ready(function () {
        
        // Sticky Menu
        $(window).scroll(function() {
            var Width = $(document).width();

            if ($("body").scrollTop() > 100 || $("html").scrollTop() > 100) {
                if (Width > 767) {
                    $(".header_2").addClass("stop");
                }
            } else {
                $(".header_2").removeClass("stop");
            }
        });

         // Sticky Menu
         $(window).scroll(function() {
            var Width = $(document).width();

            if ($("body").scrollTop() > 100 || $("html").scrollTop() > 100) {
                if (Width > 767) {
                    $(".header_3").addClass("stop");
                }
            } else {
                $(".header_3").removeClass("stop");
            }
        });

        /*** Search bar */
        $('.header-search').on('click', '.search-toggle', function(e) {
            e.preventDefault();
            var selector = $(this).data('selector');
            $(selector).toggleClass('show').find('.search-input').focus();
        });
        
        // var catButton = '.portfolio-cat-filter button';

        // $(catButton).on('click', function(){
        //     $(catButton).removeClass('active');
        //     $(this).addClass('active');
        // });


        $('#hamburger').on('click', function() {            
            $('.mobile-nav').addClass('show');
            $('.overlay').addClass('active');
        });

        $('.close-nav').on('click', function() {            
            $('.mobile-nav').removeClass('show');            
            $('.overlay').removeClass('active');          
        });

        $(".overlay").on("click", function () {
            $(".mobile-nav").removeClass("show");
            $('.overlay').removeClass('active');
        });

        $("#mobile-menu").metisMenu();

        $(".single_progress_bar span").each(function () {
            $(this).animate(
              {
                width: $(this).attr("data-progress") + "%",
              },
              1000
            );
          });

        /* =============================================
            # owlCarousel init
        ===============================================*/

        $(".hero_slider").owlCarousel({        
            items: 1,     
            dots: true,
            loop: true,
            autoplayTimeout: 8000,
            autoplay:true,
            nav: true,
            animateOut: 'fadeOut',         
            navText: ['<i class="far fa-long-arrow-right"></i>', '<i class="far fa-long-arrow-left"></i>'],
           
        });

        const $owlCarousel = $(".agency-slider-active").owlCarousel({        
            items: 1,     
            dots: true,
            loop: true,
            autoplayTimeout: 9000,
            autoplay:true,
        });

        $(".owl-carousel").on("initialized.owl.carousel", () => {
            setTimeout(() => {
                $(".owl-item.active .animated-text").addClass("is-transitioned");
            }, 200);
        });

        $owlCarousel.on("changed.owl.carousel", e => {
            $(".animated-text").removeClass("is-transitioned");
            
            const $currentOwlItem = $(".owl-item").eq(e.item.index);
            $currentOwlItem.find(".animated-text").addClass("is-transitioned");
            
            const $target = $currentOwlItem.find(".hero-contents");
        });

        $(".brand_widget").owlCarousel({        
            items: 6,     
            dots: false,
            loop: true,
            autoplayTimeout: 8000,
            autoplay:true,
            nav: false,
            loop: true,
            responsive: {
                0: {
                    items:1
                },
                500: {
                    items:2
                },
                900: {
                    items:3
                },
                1100: {
                    items:5
                },
                1200: {
                    items:6
                }

            }
        });
        $(".testimonials").owlCarousel({        
            items: 6,     
            dots: false,
            loop: true,
            autoplayTimeout: 8000,
            autoplay: true,
            margin:25,
            nav: false,
            loop: true,
            responsive: {
                0: {
                    items:1
                },
                768: {
                    items:1
                },
                900: {
                    items:2
                },
                1100: {
                    items:2
                },
                1200: {
                    items:2
                }

            }
        });
        /* =============================================
            # slick init
         ===============================================*/
         $('.single_testimonial_right_2').slick({
            dots: true,
            nav: true,
            autoplay: true,
            autoplayTimeout: 2000,
            arrows: true,
            infinite: true,
            prevArrow:"<button type='button' class='slick-prev pull-left'><i class='far fa-long-arrow-left'></i> prev</button>",
            nextArrow:"<button type='button' class='slick-next pull-right'>Next<i class='far fa-long-arrow-right'></i></button>",
            speed: 300,
            slidesToShow: 2,
            slidesToScroll: 1,
             responsive: [
            {
                breakpoint: 1080,
                settings: {
                slidesToShow: 2
                }
            },
            {
                breakpoint: 768,
                settings: {
                slidesToShow: 1
                }
            },
            {
                breakpoint: 480,
                settings: {
                slidesToShow: 1
                }
            }
            ]
          });

        /* =============================================
            # Tab Filter  init
         ===============================================*/
         $('.tab-btn').click(function() {
            // Remove active class from currently active button and content
            $('.tab-btn').removeClass('active');
            $('.faq_widget div').removeClass('active');
            
            // Add active class to clicked button and corresponding content
            $(this).addClass('active');
            $('.' + $(this).attr('id')).addClass('active');
            });

        /* =============================================
            # Magnific popup init
         ===============================================*/
         $('.popup-youtube').magnificPopup({
            type: 'iframe'
          });
        
        
        /*==========================
           Scroll To Up Init
        ============================*/
        $.scrollUp({
            scrollName: 'scrollUp',
            topDistance: '1110',
            topSpeed: 2000,
            animation: 'slide',
            animationInSpeed: 300,
            animationOutSpeed: 300,
            scrollText: '<i class="fal fa-angle-up"></i>',
            activeOverlay: false, 
        });

        new WOW().init();

        // $('.side-menu-toggle, .offcanvas-btn').on('click', function() {            
        //     $('.offset-menu').addClass('show');
        // });

        // $('#offset-menu-close-btn').on('click', function() {            
        //     $('.offset-menu').removeClass('show');
        // });

        // $('.search-btn').on('click', function() {            
        //     $('.search-box').toggleClass('show');
        // });


        $('.counter').counterUp({
            delay: 10,
            time: 3000
        });
        
    }); // end document ready function

    function loader() {
        $(window).on('load', function() {
            // Animate loader off screen
            $(".preloader").addClass('loaded');                    
            $(".preloader").delay(600).fadeOut();                       
        });
    }
    loader();

})(jQuery); // End jQuery
