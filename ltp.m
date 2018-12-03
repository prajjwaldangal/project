function [ ltp_upper, ltp_lower ] = LTP(im, t)

    %// Get the dimensions
    rows=size(im,1);
    cols=size(im,2);

    %// Reordering vector - Essentially for getting binary strings
    reorder_vector = [8 7 4 1 2 3 6 9];

    %// For the upper and lower LTP patterns
    ltp_upper = zeros(size(im));
    ltp_lower = zeros(size(im));

    %// For each pixel in our image, ignoring the borders...
    for row = 2 : rows - 1
        for col = 2 : cols - 1
            cen = im(row,col); %// Get centre

            %// Get neighbourhood - cast to double for better precision
            pixels = double(im(row-1:row+1,col-1:col+1));

            %// Get ranges and determine LTP
            out_LTP = zeros(3, 3);
            low = cen - t;
            high = cen + t;
            out_LTP(pixels < low) = -1;
            out_LTP(pixels > high) = 1;
            out_LTP(pixels >= low & pixels <= high) = 0;

            %// Get upper and lower patterns
            upper = out_LTP;
            upper(upper == -1) = 0;
            upper = upper(reorder_vector);

            lower = out_LTP;
            lower(lower == 1) = 0;
            lower(lower == -1) = 1;
            lower = lower(reorder_vector);

            %// Convert to a binary character string, then use bin2dec
            %// to get the decimal representation
            upper_bitstring = char(48 + upper);
            ltp_upper(row,col) = bin2dec(upper_bitstring);

            lower_bitstring = char(48 + lower);
            ltp_lower(row,col) = bin2dec(lower_bitstring);
       end
   end