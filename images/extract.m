% 从图像中提取两个最大目标的中心坐标
% 参数:
%   currentFrame - 当前处理的帧图像
%   backgroundImage - 背景图像
%   figMain - 主显示窗口句柄
%   figProcess - 处理过程显示窗口句柄
%   figBinary - 二值化图像显示窗口句柄
%   figForeground - 前景图像显示窗口句柄
%   frameIndex - 当前帧索引
% 返回:
%   centerX1, centerY1 - 第一个目标的中心坐标
%   centerX2, centerY2 - 第二个目标的中心坐标
%   isValid - 提取是否成功的标志
function [centerX1, centerY1, centerX2, centerY2, isValid] = extract(currentFrame, backgroundImage, figMain, figProcess, figBinary, figForeground, frameIndex)
    % 初始化返回值
    centerX1 = 0; centerY1 = 0;
    centerX2 = 0; centerY2 = 0;
    isValid = false;
    
    % 获取图像尺寸
    [imageHeight, imageWidth, ~] = size(backgroundImage);
    
    % 计算前景图像（当前帧与背景的差异）
    foreground = imsubtract(currentFrame, backgroundImage);
    
    % 显示前景图像
    if figForeground > 0
        figure(figForeground);
        clf;
        imshow(foreground);
        title('前景图像');
    end
    
    % 图像二值化处理
    binaryImage = im2bw(foreground, 45/255);
    
    % 形态学膨胀操作，连接可能的目标区域
    processedImage = bwmorph(binaryImage, 'dilate', 5);
    
    % 显示处理后的二值图像
    if figProcess > 0
        figure(figProcess);
        clf;
        imshow(processedImage);
        title('处理后的二值图像');
    end
    
    % 标记连通区域
    labeledRegions = bwlabel(processedImage, 4);
    
    % 计算区域属性
    regionStats = regionprops(labeledRegions, 'Area', 'Centroid');
    numRegions = length(regionStats);
    
    % 检查是否有足够的区域
    if numRegions < 1
        return;
    end
    
    % 对区域按面积从大到小排序
    [~, sortedIndices] = sort([regionStats.Area], 'descend');
    
    % 确保最大区域足够大
    if regionStats(sortedIndices(1)).Area < 50
        return;
    end
    
    % 获取最大区域的质心
    largestRegion = regionStats(sortedIndices(1));
    centerX1 = largestRegion.Centroid(1);
    centerY1 = largestRegion.Centroid(2);
    
    % 检查是否有第二个足够大的区域
    if numRegions > 1 && regionStats(sortedIndices(2)).Area >= 30
        secondRegion = regionStats(sortedIndices(2));
        centerX2 = secondRegion.Centroid(1);
        centerY2 = secondRegion.Centroid(2);
        isValid = true;
    else
        return;
    end
    
    % 根据帧索引交换目标顺序（可能是基于特定场景的处理逻辑）
    if frameIndex < 110
        % 在帧110之前，确保第一个目标在右侧
        if centerX1 < centerX2
            % 交换两个目标的坐标
            temp = centerX1; centerX1 = centerX2; centerX2 = temp;
            temp = centerY1; centerY1 = centerY2; centerY2 = temp;
        end
    else
        % 在帧110之后，确保第一个目标在左侧
        if centerX1 > centerX2
            % 交换两个目标的坐标
            temp = centerX1; centerX1 = centerX2; centerX2 = temp;
            temp = centerY1; centerY1 = centerY2; centerY2 = temp;
        end
    end
end