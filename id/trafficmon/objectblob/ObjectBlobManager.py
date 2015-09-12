from id.trafficmon.objectblob.ObjectBlob import ObjectBlob
import numpy as np

__author__ = 'Luqman'


class ObjectBlobManager(object):
    blob_map = None
    image_reference = None
    max_id = 0

    def __init__(self, contour_list, image):
        # print type(image)
        if (contour_list is None) and (image is None):
            self.blob_map = {}
        else:
            self.image_reference = image
            new_blob_list = []
            for contour in contour_list:
                new_blob = ObjectBlob(contour, image)
                new_blob_list.append(new_blob)

            self.list_evaluate(new_blob_list)

    def list_evaluate(self, blob_list):
        self.blob_map = {}
        i = self.max_id
        for blob in blob_list:
            self.blob_map[i] = blob
            i += 1
        self.max_id = i

    def get_image_reference(self):
        return self.image_reference

    def get_next_id(self):
        self.max_id += 1
        return self.max_id

    def entity_evaluate(self, blob):
        th = 20
        th_color = 25
        result = False
        for k in self.blob_map:
            cur_blob = self.blob_map[k]
            dist = cur_blob.get_blob_distance(blob)
            if (dist <= th) and (dist > 0):
                color_dist = cur_blob.get_blob_color_distance(blob)
                if color_dist <= th_color:
                    result = True
        return result

    def set_blob_map(self, blob_map):
        self.blob_map = blob_map.copy()

    def spatial_evaluation(self):
        th = 30
        th_color = 25
        merge_list = []
        removed_list = []
        blob_map_1 = self.blob_map.copy()
        blob_manager_reference = self.copy()
        for k in blob_map_1:
            cur_blob = blob_map_1[k]
            blob_map_reference = blob_manager_reference.blob_map
            for k1 in blob_map_reference:
                cur_blob_reference = blob_map_reference[k1]
                dist = cur_blob_reference.get_blob_distance(cur_blob)
                if (dist <= th) and (dist > 0):
                    color_dist = cur_blob_reference.get_blob_color_distance(cur_blob)
                    if color_dist <= th_color:
                        if k not in removed_list:
                            removed_list.append(k)
                        if k1 not in removed_list:
                            removed_list.append(k1)
                        overlap_tuple1 = k, k1
                        overlap_tuple2 = k1, k
                        if (overlap_tuple1 not in merge_list) and (overlap_tuple2 not in merge_list):
                            merge_list.append(overlap_tuple1)
        return merge_list, removed_list

    def get_contour_list(self):
        contour_list = []
        for blob in self.blob_map.values():
            contour_list.append(blob.get_contour())

        return contour_list

    def get_blob_count(self):
        return len(self.blob_map)

    def copy(self):
        new_blob_manager = ObjectBlobManager(None, None)
        new_blob_manager.blob_map = self.blob_map.copy()
        new_blob_manager.image_reference = np.copy(self.image_reference)
        new_blob_manager.max_id = self.max_id
        return new_blob_manager

    def remove_and_merge(self, remove_list, merge_list):
        new_contour_list = []

        # remove
        for k in self.blob_map:
            if k not in remove_list:
                new_contour_list.append(self.blob_map[k].get_contour())

        # merge
        for merge_idx in merge_list:
            k1, k2 = merge_idx
            blob1 = self.blob_map[k1]
            blob2 = self.blob_map[k2]
            new_blob = blob1.merge_blob(blob2, self.image_reference)
            new_contour_list.append(new_blob.get_contour())

        new_blob_manager = ObjectBlobManager(new_contour_list, self.image_reference)
        return new_blob_manager

    def temporal_evaluation(self, prev_blob_manager, current_image):
        # buat korelasi & korespondensi
        th = 12
        th_color = 25
        cur_blob = self.blob_map.copy()
        prev_blob = prev_blob_manager.blob_map.copy()
        pairs_cur_blob = {}
        pairs_prev_blob = {}
        for k in cur_blob:
            if k not in pairs_cur_blob:
                pairs_cur_blob[k] = []
            cur_blob_k = cur_blob[k]
            for k1 in prev_blob:
                if k1 not in pairs_prev_blob:
                    pairs_prev_blob[k1] = []
                prev_blob_k1 = prev_blob[k1]
                dist = prev_blob_k1.get_blob_distance(cur_blob_k)
                if (dist <= th) and (dist > 0):
                    pairs_cur_blob[k].append(k1)
                    pairs_prev_blob[k1].append(k)

        # got correlation of every blob / object
        # next: process blobs based on correlation
        # 1 to n -> n to 1 -> 1 to 1 -> 1 to 0 -> 0 to 1

        one_to_n = {}
        n_to_one = {}
        one_to_one_cur = {}
        one_to_one_prev = {}
        one_to_zero = {}
        zero_to_one = {}

        for key in pairs_cur_blob:
            cur_pair = pairs_cur_blob[key]
            if len(cur_pair) == 0:
                one_to_zero[key] = cur_pair
            elif len(cur_pair) == 1:
                one_to_one_cur[key] = cur_pair
            else:
                one_to_n[key] = cur_pair

        for key in pairs_prev_blob:
            prev_pair = pairs_prev_blob[key]
            if len(prev_pair) == 0:
                zero_to_one[key] = prev_pair
            elif len(prev_pair) == 1:
                one_to_one_prev[key] = prev_pair
            else:
                n_to_one[key] = prev_pair

        new_blob_map = {}
        destroy_keys = []

        # 1 to n
        for key in one_to_n:
            new_blob = cur_blob[key]
            ref_blob_keys = one_to_n[key]

            # # cek individual nanti diicek lagi
            # for ref_key in ref_blob_keys:
            #     ref_blob = prev_blob[ref_key]
            #

            # cek gabungan
            merged_blob = None
            key_used = 0
            for ref_key in ref_blob_keys:
                key_used = ref_key
                if merged_blob is None:
                    merged_blob = prev_blob[ref_key]
                else:
                    merged_blob = merged_blob.merge_blob(prev_blob[ref_key], prev_blob_manager.get_image_reference())
            if new_blob.is_similar(merged_blob):
                for ref_key in ref_blob_keys:
                    if ref_key != key_used:
                        destroy_keys.append(key_used)
                new_blob_map[key_used] = prev_blob[key_used].track(new_blob, self.image_reference)
            else:
                # oklusi
                for ref_key in ref_blob_keys:
                    occlusion_blob = prev_blob[ref_key].move_blob(self.image_reference)
                    new_blob_map[ref_key] = prev_blob[ref_key].track(occlusion_blob, self.image_reference)
            one_to_one_prev.pop(key, None)

        # n to 1
        for key in n_to_one:
            ref_blob = prev_blob[key]
            cur_blob_keys = n_to_one[key]

            # cek gabungan
            merged_blob = None
            for cur_key in cur_blob_keys:
                if merged_blob is None:
                    merged_blob = cur_blob[cur_key]
                else:
                    merged_blob = merged_blob.merge_blob(cur_blob[cur_key], self.get_image_reference())
            # if ref_blob.is_similar(merged_blob):
            new_blob_map[key] = ref_blob.track(merged_blob, self.image_reference)
            destroy_keys.append(key)
            # else:
            #     # oklusi berpisah
            #     destroy_keys.append(key)
            #     for cur_key in cur_blob_keys:
            #         new_key = prev_blob_manager.get_next_id()
            #         new_blob_map[new_key] = ref_blob.track(cur_blob[cur_key], self.image_reference)
            one_to_one_cur.pop(key, None)

        # 0 to 1
        for key in zero_to_one:
            new_blob_map[key] = prev_blob[key]
            # cek tempat yang sama
            blob = prev_blob[key]
            new_blob = ObjectBlob(blob.get_contour(), current_image)
            next_blob = blob.track(new_blob, self.image_reference)
            new_blob_map[key] = next_blob
            if new_blob.get_blob_color_distance(blob) > th_color:
                # destroy
                destroy_keys.append(key)
            if next_blob.get_n_frames_in_map() > 15:
                destroy_keys.append(key)
            # cek perkiraan tempat selanjutnya # coba nanti liat ini masih butuh enggak

        # 1 to 0
        for key in one_to_zero:
            ref_key = prev_blob_manager.get_next_id()
            new_blob_map[ref_key] = cur_blob[key]
            # masukkan ke dalam manager langsung

        # 1 to 1
        for key in one_to_one_cur:
            ref_key = one_to_one_cur[key][0]
            next_blob = prev_blob[ref_key].track(cur_blob[key], self.image_reference)
            new_blob_map[ref_key] = next_blob
            one_to_one_prev.pop(ref_key, None)

        for key in one_to_one_prev:
            next_key = one_to_one_prev[key][0]
            next_blob = prev_blob[key].track(cur_blob[next_key], self.image_reference)
            new_blob_map[key] = next_blob

        for dest in destroy_keys:
            new_blob_map.pop(dest, None)

        # print prev_blob_manager.max_id

        return self.set_next_blob_map(new_blob_map, prev_blob_manager.max_id)

        # print pairs_cur_blob, pairs_prev_blob

    def draw_contours(self, image, is_temporal):
        image_used = np.copy(image)
        image_result = np.copy(image)
        for k in self.blob_map:
            cur_blob = self.blob_map[k]
            image_result = cur_blob.draw(image_used, k, is_temporal)
            image_used = np.copy(image_result)
        return image_result

    def destroy_blob(self, key):
        blob = self.blob_map.pop(key, None)
        print "object no. "+str(key)+" is destroyed"
        if blob is not None:
            print "it has been in your screen for "+str(blob.get_n_frames_in_map())

    def set_next_blob_map(self, blob_map, max_id):
        next_blob_manager = self.copy()
        next_blob_manager.set_blob_map(blob_map)
        next_blob_manager.max_id = max_id
        return next_blob_manager

    def check(self):
        print "--- checking ---"
        print "number of blob: ", len(self.blob_map)
        for k in self.blob_map:
            print self.blob_map[k].get_area()