
class Key:
    def __init__(self, pos_center, size=None, label ="", highlighted = False):
        if size is None:
            size = [100, 100]
        self.posCenter = pos_center
        self.size = size
        self.label = label
        self.highlighted = highlighted

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # If the index fingertip is in the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor
            self.highlighted = True
            return self
        else:
            self.highlighted = False
            return False

    def check_active(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # If the index fingertip is in the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.highlighted = True
            return self
        else:
            self.highlighted = False
            return False