% Author:           Maximilian Kapsecker
% Description:      Extract Signs of DR in Retinal Images
% Inputarguments:   
%                   String imagePath = Path to Image Resource
%                   Integer class = Stage of DR
%                   String destinationPath = Save Destination Features
%                   Boolean plot = Plot Results
% Exemplary Use:
% resource = '/Users/Max/Desktop/retinalImages/0/';
% stage = 0;
% destination = '/Users/Max/Desktop/';
% draw = true;
% featureExtraction(resource, stage, destination, draw);

function [ MAT, rowValue ] = featureExtraction(imagePath, class, destinationPath, plot)

% Structuring Elements
SE10 = strel('square',10);
SE120 = strel('square', 120);

% Image-Labels in Folder
labels = dir(imagePath);
rowValue = struct2table(labels);
rowValue = rowValue.name;

% Number of Elements in Folder
length = size(labels,1);

% Initialze Feature Matrix
MAT = zeros(length - 3, 6);
MAT = array2table(MAT);
MAT.Properties.VariableNames(1) = {'Label'};
MAT.Properties.VariableNames(2) = {'Class'};
MAT.Properties.VariableNames(3) = {'Bloodvessel'};
MAT.Properties.VariableNames(4) = {'Exudate'};
MAT.Properties.VariableNames(5) = {'Haemorrhages'};
MAT.Properties.VariableNames(6) = {'Contrast'};

% Iterate over all Images in Folder
% On OsX Files in Folder start at 4
for i = 4:104
    
    labels(i).name
    % Load Image
    J = imread(strcat(imagePath, labels(i).name));
    haemorrhage = J;
    % Size of Image
    [width, height, components] = size(J);
    
    % Convert to GreyScale
    if(components == 3)
        J = J(:,:,2);
    end
    
    % Initialize auxiliary Images
    opticDisc = zeros(width, height);
    macula = zeros(width, height);
    binaryMask = J;
    gauss = imgaussfilt(J, 50);
    
    % Binary Mask
    binaryMask(binaryMask < 7) = 255;
    binaryMask(binaryMask < 255) = 0;
    binaryMask = imdilate(binaryMask ,SE120);
    binaryMask = im2uint8(binaryMask);
    
    gaussInv = imgaussfilt(imcomplement(J) - binaryMask, 50);
    
    % Optic Disc Detection
    [m, n] = find(gauss == max(max(gauss)));
    [~, x] = size(m);
    index = round(x / 2);
    a = m(index);
    b = n(index);
    k = round(width / 9);
    l = round(height / 9);
    for e = 1:k
        for d = 1:l
            if ((- round(k/2) + e)^2 + (- round(l/2) + d)^2) < round(3*k / 4)^2
                if(a - round(k/2) + e < width && b - round(l/2) + d < height)
                    opticDisc(a - round(k/2) + e, b - round(l/2) + d) = 255;
                end
            end
        end
    end
    opticDisc = im2uint8(opticDisc);
    
    % Macula Detection
    [m, n] = find(gaussInv == max(max(gaussInv)));
    [~, x] = size(m);
    index = round(x / 2);
    a = m(index);
    b = n(index);
    k = round(width / 10);
    l = round(height / 10);
    for e = 1:k
        for d = 1:l
            if ((- round(k/2) + e)^2 + (- round(l/2) + d)^2) < round(3*k / 4)^2
                if(a - round(k/2) + e < width && a - round(k/2) + e > 0 && ...
                    b - round(l/2) + d < height && b - round(l/2) + d > 0)
                    macula(a - round(k/2) + e, b - round(l/2) + d) = 255;
                end
            end
        end
    end
    macula = im2uint8(macula);
    
    % Blood Vessel Detection
    bloodvessel = J;
    bloodvessel = histeq(bloodvessel);
    bloodvessel = imcomplement(bloodvessel);
    bloodvessel1 = imgaussfilt(bloodvessel, 50);
    bloodvessel = bloodvessel - bloodvessel1;
    bloodvessel = imadjust(bloodvessel);
    bloodvessel = bloodvessel - binaryMask;
    bloodvessel = 2 * bloodvessel;
    %bloodvessel = im2bw(bloodvessel, 0.5);
    %bloodvessel = imerode(bloodvessel, SE5);
    %bloodvessel = imdilate(bloodvessel, SE3);
    %bloodvessel = im2uint8(bloodvessel) - macula;
    whiteBV = size(find(bloodvessel==255));
    whiteBloodvessel = whiteBV(1) / (width * height);
    
    % Exudate Detection
    exudate = imgaussfilt(J, 2);
    exudate = exudate - gauss;
    exudate = imadjust(exudate);
    exudate = imdilate(exudate, SE10);
    exudate = exudate - gauss;
    exudate = 2*exudate - opticDisc;
    %exudate = im2uint8(im2bw(exudate, 0.7));
    %exudate = im2uint8(exudate);
    %exudate = exudate - binaryMask - opticDisc;
    whiteEx = size(find(exudate==255));
    whiteExudate = whiteEx(1) / (width * height);
    whiteExudate
    
    % Haemhorrage Detection  
    haemorrhage(:,:,1) = bloodvessel;
    haemorrhage(:,:,2) = exudate;
    haemorrhage(:,:,3) = bloodvessel;
    whiteHaem = size(find(haemorrhage==255));
    whiteHaemorrhage = whiteHaem(1) / (width * height);
    
    % Contrast Determination
    glcms = graycomatrix(J,'NumLevels',256);
    contrast = graycoprops(glcms,{'contrast'});
    
    if(plot)
        
        subplot(2,2,1)
        imshow(J);
        subplot(2,2,2)
        imshow(bloodvessel);
        subplot(2,2,3)
        imshow(exudate);
        subplot(2,2,4)
        imshow(haemorrhage);
    
        w = waitforbuttonpress;
        
        if w == 0
            %MAT(i - 3, 1) = {rowValue.name(i)};
            MAT(i - 3, 2) = {class};
            MAT(i - 3, 3) = {whiteBloodvessel};
            MAT(i - 3, 4) = {whiteExudate};
            MAT(i - 3, 5) = {whiteHaemorrhage};
            MAT(i - 3, 6) = {contrast.Contrast};
            disp('Accepted')
            
        else
            
            disp('Not Accepted')
            
        end
        
    else
        
        %MAT(i - 3, 1) = {rowValue.name(i)};
        MAT(i - 3, 2) = {class};
        MAT(i - 3, 3) = {whiteBloodvessel};
        MAT(i - 3, 4) = {whiteExudate};
        MAT(i - 3, 5) = {whiteHaemorrhage};
        MAT(i - 3, 6) = {contrast.Contrast};
        
    end
    

end

% Write Feature Matrix to Disc
writetable(MAT, strcat(destinationPath, num2str(class)));

end