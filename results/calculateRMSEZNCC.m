clear all
close all
clc

% ALERT: OCTAVE
% pkg load image


listing_real     = dir('../dataset/testB/');
proposeddir      = strcat(strcat('outputs/');
listing_proposed = dir(proposeddir);

RMSE_prop_potsdam = [ ];
RMSE_prop_vaihingen = [ ];
RMSE_prop_sweden = [ ];
ZNCC_prop_potsdam = [ ];
ZNCC_prop_vaihingen = [ ];
ZNCC_prop_sweden = [ ];

for i = 1:length(listing_real)
    if length(listing_real(i).name) > 5
        filename_orig = ['../dataset/testB/',listing_real(i).name];
        filename_prop = [proposeddir,listing_proposed(i).name];
        orig = rgb2gray(imread(filename_orig));
        prop = rgb2gray(imread(filename_prop));
        rms = sqrt(immse(orig,prop));
        zn = zncc(orig,prop);

        %if ~contains(listing_real(i).name, 'potsdam') == 0
        if ~strfind(listing_real(i).name, 'potsdam') == 0
            RMSE_prop_potsdam = [RMSE_prop_potsdam rms];
            ZNCC_prop_potsdam = [ZNCC_prop_potsdam zn];
        end
        %if ~contains(listing_real(i).name, 'vaihingen') == 0
        if ~strfind(listing_real(i).name, 'vaihingen') == 0
            RMSE_prop_vaihingen = [RMSE_prop_vaihingen rms];
            ZNCC_prop_vaihingen = [ZNCC_prop_vaihingen zn];
        end
        %if ~contains(listing_real(i).name, 'sweden') == 0
        if ~strfind(listing_real(i).name, 'sweden') == 0
            RMSE_prop_sweden = [RMSE_prop_sweden rms];
            ZNCC_prop_sweden = [ZNCC_prop_sweden zn];
        end
    end
end
fprintf(proposeddir)
fprintf('\n')
fprintf('Proposed %f,%f\n',mean(RMSE_prop_potsdam)/10, std(RMSE_prop_potsdam)/10)
fprintf('Proposed %f,%f\n',mean(RMSE_prop_vaihingen)/10, std(RMSE_prop_vaihingen)/10)
fprintf('Proposed %f,%f\n',mean(RMSE_prop_sweden)/10, std(RMSE_prop_sweden)/10)

fprintf('Proposed %f,%f\n',mean(ZNCC_prop_potsdam), std(ZNCC_prop_potsdam))
fprintf('Proposed %f,%f\n',mean(ZNCC_prop_vaihingen), std(ZNCC_prop_vaihingen))
fprintf('Proposed %f,%f\n',mean(ZNCC_prop_sweden), std(ZNCC_prop_sweden))
fprintf('\n')
fprintf('\n')
