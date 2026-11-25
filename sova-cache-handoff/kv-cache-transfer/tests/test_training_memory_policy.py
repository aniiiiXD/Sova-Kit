"""Saved-tensor CPU placement must preserve the actual teacher/student loss."""
import os
import unittest
from contextlib import nullcontext


@unittest.skipUnless(os.environ.get('KVVERIFY_TORCH_SMOKE')=='1','Explicit tiny CPU smoke opt-in')
class SavedTensorPolicyTests(unittest.TestCase):
    def test_cpu_saved_tensors_preserve_loss_and_every_mapper_gradient(self):
        from test_qa_training import TinyQATrainingTests
        from kvverify.qa_training import qa_training_loss,split_document_prompt
        helper=TinyQATrainingTests();helper.setUp()
        torch=helper.torch;runtime=helper.runtime;mapper=helper.mapper()
        ids,prefix,suffix=split_document_prompt(runtime,helper.train[0])
        gradients=[];losses=[]
        for offload in (False,True):
            mapper.zero_grad(set_to_none=True)
            with (torch.autograd.graph.save_on_cpu(pin_memory=True)if offload else nullcontext()):
                loss,_=qa_training_loss(runtime,mapper,ids,prefix,suffix,answer_tokens=2,checkpoint_mapper=True)
                loss.backward()
            losses.append(loss.detach().clone())
            gradients.append({n:p.grad.detach().clone()for n,p in mapper.named_parameters()if p.grad is not None})
        torch.testing.assert_close(*losses,rtol=1e-6,atol=1e-7)
        self.assertEqual(set(gradients[0]),set(gradients[1]))
        for name in gradients[0]:torch.testing.assert_close(gradients[0][name],gradients[1][name],rtol=1e-5,atol=1e-6)
        self.assertTrue(all(p.grad is None for model in(runtime.source,runtime.target)for p in model.parameters()))


if __name__=='__main__':unittest.main()
