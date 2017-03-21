function [data_clusters, cluster_stats] = my_kmeans( data, nc )

% This function performs k-means clustering on data ,   given (nc) = the number of clusters.


%  Random Initialization

ndata = size(data,1);
ndims = size(data,2);

random_labels = floor(rand(ndata,1) * nc) + 1;

data_clusters = random_labels;

cluster_stats = zeros(nc,ndims+1);

distances = zeros(ndata,ndims);

while(1)
    
    pause(0.03);
    
    % Make a copy of cluster statistics for 
    % comparison purposes.  If the difference is very small, the while loop will exit.
    last_clusters = cluster_stats;
    
    % For each cluster    
    for c=1:nc
        
        % Find all data points assigned to this cluster
        [ind] = find(data_clusters == c);
        num_assigned = size(ind,1);
        
        % some heuristic codes for exception handling. 
        if( num_assigned < 1 )
            disp('No points were assigned to this cluster, some special processing is given below');
            
            % Calculate the maximum distances from each cluster
            max_distances = max(distances);
            
            [maxx,cluster_num] = max(max_distances);
            [maxx,data_point] = max(distances(:,cluster_num));
            
            data_clusters(data_point) = cluster_num;
            
            ind = data_point;
            num_assigned = 1;
        end   %% end of exception handling.   
        
        % Save number of points per cluster,  plus the mean vectors.
        cluster_stats(c,1) = num_assigned;
        if( num_assigned > 1 )
            summ = sum(data(ind,:));
            cluster_stats(c,2:ndims+1) = summ / num_assigned;
        else
            cluster_stats(c,2:ndims+1) = data(ind,:);
        end
        
    end
    
    % Exit criteria
    diff = sum(abs(cluster_stats(:) - last_clusters(:)));
    if( diff < 0.00001 )
        break;
    end
    
    % - Set each cluster center to the average of the points assigned to it.
    
    % - Assign each point to the nearest cluster center  
    ... FILL your code here...
	
	%%update the membership assignment, i.e., update the data_clusters with current values.  
    
	...FILL your code here... 
	
    
    % Display clusters for the purpose of debugging.  
    cluster_stats
    %pause;
 end 
% for example:  cluster_stats =
% 
%   172.0000   -9.4800
%   177.0000    5.3816
%   192.0000   -2.1585 

% cluster_stats =
% 
%   165.0000   -9.6324
%   181.0000    5.2970
%   195.0000   -2.3686
