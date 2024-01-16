function I_blur = gaussian(I, sigma)
    % Applies a 2-D Gaussian blur with standard deviation sigma to
    % a grayscale image I.

    % Hint: The size of the kernel should depend on sigma. A common
    % choice is to make the half-width be 3 standard deviations. The
    % total kernel width is then 2*ceil(3*sigma) + 1.

    I_blur = zeros(size(I)); % Placeholder
end
