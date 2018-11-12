# flake8: noqa

import numpy as np

"""
function [categoryRoot,classRootId, category,classId,insctancId] = getobjclassSUNCG(objname,objcategory)
            [~,insctancId] = ismember(objname,objcategory.all_labeled_obj);
            if (insctancId>0)
                classId = objcategory.classid(insctancId);
                category = objcategory.allcategories{objcategory.classid(insctancId)};
            else
                category = objname;
                classId = -1;
            end
            classRootId = length(objcategory.object_hierarchical)+1;
            categoryRoot = 'dont_care';
            for i = 1:length(objcategory.object_hierarchical)
                if ( ismember(category,objcategory.object_hierarchical{i}.clidern))
                    classRootId =i;
                    categoryRoot = objcategory.object_hierarchical{i}.categoryname;
                    break;
                end
            end
end
"""


def getobjclassSUNCG(objname, objcategory):
    try:
        instanceId = objcategory.all_labeled_obj.tolist().index(objname)
    except ValueError:
        instanceId = -1
    if instanceId >= 0:
        classId = objcategory.classid[instanceId]
        category = objcategory.allcategories[classId - 1]
    else:
        classId = -1
        category = objname
    classRootId = len(objcategory.object_hierarchical) + 1
    categoryRoot = 'dont_care'
    for i in range(len(objcategory.object_hierarchical)):
        if category in objcategory.object_hierarchical[i].clidern:
            classRootId = i + 1
            categoryRoot = objcategory.object_hierarchical[i].categoryname
            break
    return categoryRoot, classRootId, category, classId, instanceId
