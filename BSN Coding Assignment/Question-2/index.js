/* ======================= */
/* DO NOT CHANGE THE BELOW */
/* ======================= */
var PRODUCTS = [
    {
        'name'        : 'Google Pixel 3 XL Unlocked GSM/CDMA',
        'description' : 'Create, share and stay connected with this black or white Google Pixel 3 XL smartphone. Its 64GB of ' +
                        'storage lets you save important files and apps, and the 12.2-megapixel rear camera has autofocus to take' +
                        'professional-looking photos easily. The 6.3-inch touch screen on this Google Pixel 3 XL smartphone is' +
                        'water-resistant and dust-resistant.',
        'price'       : 899.00,
        'image'       : 'https://images-na.ssl-images-amazon.com/images/I/81LvcrVzWKL._SL1500_.jpg',
        'review'      : {
            'rating'  : 4.8,
            'count'   : 100
        }
    },
    {
        'name'        : 'Apple iPhone X (64GB) Silver',
        'description' : '5.8-inch Super Retina display (OLED) with HDR. 12MP dual cameras with dual OIS and 7MP True Depth ' +
                        'front camera: Portrait mode and Portrait Lighting. Face ID for secure authentication and Apple Pay. ' +
                        'A11 Bionic with Neural Engine. Wireless charging works with Qi chargers',
        'price'       : 899.00,
        'image'       : 'https://images-na.ssl-images-amazon.com/images/I/51R4ZvEJUPL._SL1024_.jpg',
        'review'      : {
            'rating'  : 4.5,
            'count'   : 120
        }
    },
    {
        'name'        : 'Samsung Galaxy S9 with 256GB Memory Cell Phone',
        'description' : 'Enjoy the ultimate mobile experience whether calling or scrolling the internet with this ' +
                        'Samsung Galaxy S9 smartphone. The super-speed dual 12MP rear cameras provide crisp, stabilized ' +
                        'images that look great on and off the Super AMOLED screen. This slim Samsung Galaxy S9letsyou run ' +
                        'multiple apps simultaneously thanks to 256GB of RAM built inside.',
        'price'       : 839.00,
        'image'       : 'https://pisces.bbystatic.com/image2/BestBuy_US/images/products/6256/6256613_sd.jpg;maxHeight=1000;maxWidth=1000',
        'review'      : {
            'rating'  : 4.2,
            'count'   : 28
        }
    },
    {
        'name'        : 'LG V35 ThinQ with 64GB Memory Cell Phone',
        'description' : 'Improve your picture-taking with this LG V35 unlocked smartphone. Its dual 16.0-megapixel camera '+
                        'provides high-resolution wide-angle photography, and the AI Cam feature analyzes your view to help ' +
                        'frame the perfect shots. This LG V35 unlocked smartphone includes 64GB of internal storage and ' +
                        'supports a microSD card up to 2TB to store all your photos, apps and data.',
        'price'       : 854.99,
        'image'       : 'https://pisces.bbystatic.com/image2/BestBuy_US/images/products/6282/6282528_sd.jpg;maxHeight=1000;maxWidth=1000',
        'review'      : {
            'rating'  : 4.3,
            'count'   : 34
        }
    }
];

/* ======================= */
/* DO NOT CHANGE THE ABOVE */
/* ======================= */

/* ===================== */
/* WRITE YOUR CODE BELOW */
/* ===================== */

function generateProductList() {
    var input, filter;
    input = document.getElementById("searchText");
    filter = input.value.toLowerCase();
    li  = document.getElementById("results").getElementsByTagName('li');

    var i;
    var noItem = true;
    for(i=0; i<PRODUCTS.length; i++) {
        if(PRODUCTS[i].name.toLowerCase().indexOf(filter) > -1) {
            li[i].style.display = "";
            noItem = false;
        } else {
            li[i].style.display = "none";
        }
    }
    if (noItem) {
        var nothing = document.createElement("li");
        nothing.setAttribute("id", "no-item");
        nothing.innerHTML = "No item found.";
        document.getElementById("results").appendChild(nothing);
    } else {
        try {
            document.getElementById("results").removeChild(document.getElementById("no-item"));
        }
        catch(err) {};
    }
}