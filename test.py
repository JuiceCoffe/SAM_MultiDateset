                self.category_overlapping_mask_teacher, self.train_num_templates_teacher, self.train_class_names_teacher = self.prepare_class_names_from_metadata(self.train_metadata[dataname], self.train_metadata, prompt_teacher)
                text_classifier = []
                bs = 128
                print("Generating text classifier for", dataname, "with", len(self.train_class_names), "classes.")
                for idx in range(0, len(self.train_class_names_teacher), bs):
                    text_classifier.append(self.backbone2.get_text_classifier(self.train_class_names_teacher[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)
                print("text_classifier shape before normalize:", text_classifier.shape)

                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                print("text_classifier shape before reshape:", text_classifier.shape)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(prompt_teacher), len(prompt_teacher), text_classifier.shape[-1]).mean(1) 
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.text_classifier2 = text_classifier
                self.train_dataname = dataname
            return self.text_classifier2, self.train_num_templates_teacher