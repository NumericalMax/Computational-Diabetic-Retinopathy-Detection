% Author:           Maximilian Kapsecker
% Description:      PreProcess Retinal Images, for CNN training
% Inputarguments:   String imagePath = Path to Image Resource
%                   String destinationPath = Save Destination Features
% Exemplary Use:    resource = '/Users/Max/Desktop/retinalImages/0/';
%                   destination = '/Users/Max/Desktop/preProcessedImages/0/';
%                   featureExtraction(resource, stage, destination, draw);

function [] = preProcessing(imagePath, destinationPath)

    % Structuring Elements
    SE = strel('square', 5);

    % Image-Labels in Folder
    labels = dir(imagePath);

    % Number of Elements in Folder
    length = size(labels, 1);

    % Iterate over all Images in Folder
    % On OsX Files in Folder start at 4
    for i = 4:length
    
        labels(i).name
    
        % Load Image
        J = imread(strcat(imagePath, labels(i).name));
        L = J;
    
        [~,~, components] = size(J);
    
        % Convert to GreyScale if necessary
        if(components == 3)
            J = J(:,:,2);
        end
        
        % Determine Binary Mask
        Q = J;
        Q(Q < 20) = 255;
        Q(Q < 255) = 0;
        Q = imdilate(Q, SE);
        
        % Visualization of  Haemhorrhages and BloodVessels (R-Component)
        %                   Exudate (G-Component)
        %                   Microaneurysm (B-Component)
        
        I = J;
        J = imcomplement(J);
    
        I = I - imgaussfilt(I, 10);
        K = J - imgaussfilt(J, 40);
        J = J - imgaussfilt(J, 10);

        I = imerode(I, SE);
        J = imerode(J, SE);
        K = imerode(K, SE);

        L(:,:,1) = K - Q;
        L(:,:,2) = I - Q;
        L(:,:,3) = J - Q;
    
        L = 15*L;
        
        % Save Processed Image back to Disc
        imwrite(L, strcat(destinationPath, labels(i).name));
        
    end

end