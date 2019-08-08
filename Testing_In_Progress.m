%% Read image    
 
    [Igray,map,alpha] = imread('480nm_1.png'); % Read image
    [Pixel1, Pixel2] = size(Igray);                     % Get size of image (for experimental purposes)
    imshow(Igray, map);                                % Show original image to make sure image is read correctly

    [rows, columns, numberOfColorChannels] = size(Igray); 
    if numberOfColorChannels > 1
        Igray = rgb2gray(Igray);               % Convert rgb to gray
    else
        Igray = rgb2gray(map);
    end

    %figure(2);
    %imshow(Igray);

    %% Preliminary Crop

    [Pixel3, Pixel4] = size(Igray);        % Gets size of grayed image
    CropBox = 700;                         % Size of preliminary crop box
    x_start = floor((Pixel3 - CropBox)/2); % Min x value for crop
    x_stop = x_start + CropBox;            % Max x value for crop 
    y_start = floor((Pixel4 - CropBox)/2); % Min y value for crop
    y_stop = y_start + CropBox;            % Max y value for crop
    NewI = Igray(floor(x_start:x_stop), floor(y_start:y_stop),:); % Crops the grayed image
    figure;
    imshow(NewI)
    NewII = imgaussfilt(NewI,4);
    figure;
    imshow(NewI)
    J = edge(NewII,'Canny',0.2);
    imshow(J)
%     figure; imshow(J)
%     J = bwareaopen(J,4);
%     figure; imshow(J);
%     se = strel('disk',20);
%     J = imclose(J,se);
%     J = bwmorph(J,'remove');
%     figure; imshow(J);
%     J = bwareaopen(J,10);
%     J = imdilate(J,strel('disk',3));
%     J = imerode(J,strel('disk',2));
%     conn = [0 1 0; 0 1 0; 0 1 0;];
%     %J = imfill(J,8,'holes');
%     figure;
%     imshow(J);
    
%% Circle Detection
    tic
    [centers,radii] = imfindcircles(J, [100 200],'ObjectPolarity','bright',...
        'Sensitivity', 0.94,'Method','twostage');       % Finds the large test fringe (experimental)
    radii = radii + 20;             % Accounting for very dark fringes,  captured by line above
    h = viscircles(centers,radii);  % Region of circle that will be cropped
    toc
    %% Final Crop

    x_start1 = centers(1,2) - radii;        % Min x value for crop
    x_stop1 = x_start1 + 2*radii;         % Max x value for crop
    y_start1 = centers(1,1) - radii;       % Min y value for crop
    y_stop1 = y_start1 + 2*radii;         % Max y value for crop
    NewI2 = NewI(floor(x_start1:x_stop1),floor(y_start1:y_stop1));% Crops desired image
    SizeNewI2 = imresize(NewI2,3);      % Resize the image to 3 times its size
    figure; imshow(SizeNewI2);
    gCounter = 4;
    SizeNewII2 = imgaussfilt(SizeNewI2,gCounter);
    SizeNewII2 = edge(SizeNewII2,'Canny',0.37);
    SizeNewII2 = imdilate(SizeNewII2,strel('disk',3));
    SizeNewII2 = imerode(SizeNewII2,strel('disk',2));
    SizeNewII2 = imdilate(SizeNewII2,strel('disk',3));
    figure; imshow(SizeNewII2);
    [centers,radii] = imfindcircles(SizeNewII2, [380 400],'ObjectPolarity','bright',...
        'Sensitivity', 0.97,'Method','twostage');
    if size(radii,1) == 0
        while(size(radii,1) == 0)
            gCounter = gCounter - 0.1;
            SizeNewII2 = imgaussfilt(SizeNewI2,gCounter - 0.1);
            SizeNewII2 = edge(SizeNewII2,'Canny',0.37);
            SizeNewII2 = imdilate(SizeNewII2,strel('disk',3));
            SizeNewII2 = imerode(SizeNewII2,strel('disk',2));
            SizeNewII2 = imdilate(SizeNewII2,strel('disk',3));
            figure; imshow(SizeNewII2);
            [centers,radii] = imfindcircles(SizeNewII2, [380 400],'ObjectPolarity','bright',...
                'Sensitivity', 0.97,'Method','twostage');
        end
    end
    h = viscircles(centers(1,:),radii(1));
    x_start2 = centers(1,2) - radii(1);        
    x_stop2 = x_start2 + 2*radii(1);         
    y_start2 = centers(1,1) - radii(1);       
    y_stop2 = y_start2 + 2*radii(1);
    counter = 2;
if (x_start2 < 0  || x_stop2 > size(SizeNewII2,1) || y_start2 < 0 || y_stop2 > size(SizeNewII2,1))
    while(x_start2 < 0  || x_stop2 > size(SizeNewII2,1) || y_start2 < 0 || y_stop2 > size(SizeNewII2,1))
        x_start2 = centers(counter,2) - radii(counter);        
        x_stop2 = x_start2 + 2*radii(counter);         
        y_start2 = centers(counter,1) - radii(counter);       
        y_stop2 = y_start2 + 2*radii(counter);
        counter = counter + 1;
    end
end
    NewI3 = SizeNewI2(floor(x_start2:x_stop2),floor(y_start2:y_stop2));
    figure(3); imshow(NewI3);
%     ortho_points = [centers(1),centers(2);centers(1), centers(2) + radii; centers(1) + radii, centers(2);...
%         centers(1), centers(2) - radii; centers(1) - radii, centers(2)];
%     aerial_points = [centers(1),centers(2);centers(1), centers(2) + radii; centers(1) + radii, centers(2);...
%         centers(1), centers(2) - radii; centers(1) - radii, centers(2)];
%     saveas(figure(2),fullfile('C:\Users\Andrew\Documents\MATLAB','Red_Bad_Big.jpg'))
%     saveas(figure(4),fullfile('C:\Users\Andrew\Documents\MATLAB','Red_Bad__Small.jpg'))