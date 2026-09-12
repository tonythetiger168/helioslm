"""K3+ v4.0 Test Suite"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import torch
from src.configs import ultra_config, pro_config, lite_config, nano_config
from src.model import KimiK3Plus
from src.attention import KimiDeltaAttention
from src.moe import StableLatentMoE
from src.speculative_decoding import DraftModel, SpeculativeDecoder
from src.rag import RAGModule
from src.cot_compiler import CoTCompiler
from src.reasoning_budget import ReasoningBudgetController
from src.agentic import AgenticLayer
from src.memory import LongTermMemory
from src.multimodal import MultimodalFusion, VisionEncoder, AudioEncoder
from src.adaptive_reasoning_v2 import AdaptiveReasoningController, MCTSReasoningSearch
from src.online_learning import UserLoRA, OnlineLearningManager
from src.safety_alignment import SafetyAlignmentLayer, HarmfulnessClassifier
from src.compression import CompressionManager, KVCacheCompressor, DynamicQuantizer

def test_sizes():
    for name,cfg in [("ultra",ultra_config),("pro",pro_config),("lite",lite_config),("nano",nano_config)]:
        m=KimiK3Plus(cfg,size=name); print(f"  {name.upper()}: {sum(p.numel() for p in m.parameters()):,} params")
    print("✅ 4 sizes OK")

def test_forward():
    m=KimiK3Plus(lite_config,size="lite")
    ids=torch.randint(0,lite_config.vocab_size,(2,128))
    logits,aux=m(ids)
    assert logits.shape==(2,128,lite_config.vocab_size) and aux>=0
    print("✅ Forward OK")

def test_speculative():
    m=KimiK3Plus(lite_config,size="lite")
    draft=DraftModel(lite_config)
    dec=SpeculativeDecoder(m,draft,lite_config)
    ids=torch.randint(0,lite_config.vocab_size,(1,10))
    out,rate,speed=dec.generate(ids,20)
    assert out.shape[1]>ids.shape[1]
    print(f"✅ Speculative OK (rate:{rate:.2f}, speed:{speed:.1f})")

def test_rag():
    rag=RAGModule(lite_config)
    h=torch.randn(2,10,lite_config.hidden_size)
    out=rag(h)
    assert "confidence" in out
    print("✅ RAG OK")

def test_cot():
    cot=CoTCompiler(lite_config)
    h=torch.randn(1,10,lite_config.hidden_size)
    out=cot.compile_reasoning(h)
    assert "max_steps" in out
    print("✅ CoT OK")

def test_budget():
    rb=ReasoningBudgetController(lite_config)
    h=torch.randn(1,10,lite_config.hidden_size)
    budget,diff=rb.get_budget(h,4096)
    assert budget>0
    print(f"✅ Budget OK (budget:{budget}, diff:{diff})")

def test_agentic():
    a=AgenticLayer(lite_config)
    h=torch.randn(1,10,lite_config.hidden_size)
    out=a(h)
    assert "tools" in out
    print("✅ Agentic OK")

def test_memory():
    mem=LongTermMemory(lite_config)
    k=torch.randn(lite_config.hidden_size)
    v=torch.randn(100,lite_config.hidden_size)
    mem.add(k,v,1.5)
    r=mem.retrieve(k,3)
    assert len(r)>0
    print("✅ Memory OK")

def test_multimodal():
    mm=MultimodalFusion(lite_config)
    h=torch.randn(1,10,lite_config.hidden_size)
    out=mm(h)
    assert out.shape==h.shape
    print("✅ Multimodal OK")

def test_adaptive_reasoning():
    ar=AdaptiveReasoningController(lite_config)
    h=torch.randn(1,10,lite_config.hidden_size)
    budget,level=ar.get_budget(h)
    assert budget>0
    print(f"✅ Adaptive Reasoning OK (level:{level}, budget:{budget})")

def test_online_learning():
    ol=OnlineLearningManager(lite_config)
    adapter=ol.get_user_adapter("test_user")
    assert adapter is not None
    print("✅ Online Learning OK")

def test_safety():
    safety=SafetyAlignmentLayer(lite_config)
    h=torch.randn(1,10,lite_config.hidden_size)
    out=safety(h)
    assert "safety" in out
    print("✅ Safety Alignment OK")

def test_compression():
    comp=CompressionManager(lite_config)
    k=torch.randn(1,4,100,lite_config.hidden_size)
    v=torch.randn(1,4,100,lite_config.hidden_size)
    ck,cv=comp.compress_kv_cache(k,v,100)
    assert ck.shape[2]<k.shape[2]
    print(f"✅ Compression OK ({k.shape[2]} -> {ck.shape[2]} tokens)")

if __name__=="__main__":
    print("🧪 K3+ v4.0 Test Suite"); print("="*40)
    test_sizes(); test_forward(); test_speculative(); test_rag(); test_cot(); test_budget(); test_agentic(); test_memory()
    test_multimodal(); test_adaptive_reasoning(); test_online_learning(); test_safety(); test_compression()
    print("="*40+"\n🎉 All tests passed!")
