% ��ͼ������ȡ�������Ŀ�����������
% ����:
%   currentFrame - ��ǰ�����֡ͼ��
%   backgroundImage - ����ͼ��
%   figMain - ����ʾ���ھ��
%   figProcess - ���������ʾ���ھ��
%   figBinary - ��ֵ��ͼ����ʾ���ھ��
%   figForeground - ǰ��ͼ����ʾ���ھ��
%   frameIndex - ��ǰ֡����
% ����:
%   centerX1, centerY1 - ��һ��Ŀ�����������
%   centerX2, centerY2 - �ڶ���Ŀ�����������
%   isValid - ��ȡ�Ƿ�ɹ��ı�־
function [centerX1, centerY1, centerX2, centerY2, isValid] = extract(currentFrame, backgroundImage, figMain, figProcess, figBinary, figForeground, frameIndex)
    % ��ʼ������ֵ
    centerX1 = 0; centerY1 = 0;
    centerX2 = 0; centerY2 = 0;
    isValid = false;
    
    % ��ȡͼ��ߴ�
    [imageHeight, imageWidth, ~] = size(backgroundImage);
    
    % ����ǰ��ͼ�񣨵�ǰ֡�뱳���Ĳ��죩
    foreground = imsubtract(currentFrame, backgroundImage);
    
    % ��ʾǰ��ͼ��
    if figForeground > 0
        figure(figForeground);
        clf;
        imshow(foreground);
        title('ǰ��ͼ��');
    end
    
    % ͼ���ֵ������
    binaryImage = im2bw(foreground, 45/255);
    
    % ��̬ѧ���Ͳ��������ӿ��ܵ�Ŀ������
    processedImage = bwmorph(binaryImage, 'dilate', 5);
    
    % ��ʾ�����Ķ�ֵͼ��
    if figProcess > 0
        figure(figProcess);
        clf;
        imshow(processedImage);
        title('�����Ķ�ֵͼ��');
    end
    
    % �����ͨ����
    labeledRegions = bwlabel(processedImage, 4);
    
    % ������������
    regionStats = regionprops(labeledRegions, 'Area', 'Centroid');
    numRegions = length(regionStats);
    
    % ����Ƿ����㹻������
    if numRegions < 1
        return;
    end
    
    % ����������Ӵ�С����
    [~, sortedIndices] = sort([regionStats.Area], 'descend');
    
    % ȷ����������㹻��
    if regionStats(sortedIndices(1)).Area < 50
        return;
    end
    
    % ��ȡ������������
    largestRegion = regionStats(sortedIndices(1));
    centerX1 = largestRegion.Centroid(1);
    centerY1 = largestRegion.Centroid(2);
    
    % ����Ƿ��еڶ����㹻�������
    if numRegions > 1 && regionStats(sortedIndices(2)).Area >= 30
        secondRegion = regionStats(sortedIndices(2));
        centerX2 = secondRegion.Centroid(1);
        centerY2 = secondRegion.Centroid(2);
        isValid = true;
    else
        return;
    end
    
    % ����֡��������Ŀ��˳�򣨿����ǻ����ض������Ĵ����߼���
    if frameIndex < 110
        % ��֡110֮ǰ��ȷ����һ��Ŀ�����Ҳ�
        if centerX1 < centerX2
            % ��������Ŀ�������
            temp = centerX1; centerX1 = centerX2; centerX2 = temp;
            temp = centerY1; centerY1 = centerY2; centerY2 = temp;
        end
    else
        % ��֡110֮��ȷ����һ��Ŀ�������
        if centerX1 > centerX2
            % ��������Ŀ�������
            temp = centerX1; centerX1 = centerX2; centerX2 = temp;
            temp = centerY1; centerY1 = centerY2; centerY2 = temp;
        end
    end
end