// SPDX-License-Identifier: Apache-2.0
package apb_portfolio_test_pkg;
    import uvm_pkg::*;
    import amba_vip_pkg::*;
    `include "uvm_macros.svh"

    class portfolio_sequence extends uvm_sequence #(amba_seq_item);
        `uvm_object_utils(portfolio_sequence)
        function new(string name = "portfolio_sequence"); super.new(name); endfunction

        task transfer(bit wr, bit [31:0] address, bit [31:0] data,
                      resp_e expected_response, bit [31:0] expected_data);
            amba_seq_item item = amba_seq_item::type_id::create("item");
            start_item(item);
            item.txn.protocol = APB4;
            item.txn.addr = address;
            item.txn.write = wr;
            item.txn.len = 1;
            item.txn.delay = 0;
            item.txn.prot = 0;
            item.txn.data = new[1];
            item.txn.data[0] = {32'b0, data};
            finish_item(item);
            if (item.txn.resp !== expected_response ||
                (!wr && item.txn.data[0][31:0] !== expected_data))
                `uvm_fatal("APB_DATA", $sformatf("unexpected response at %h: %h, data %h",
                          address, item.txn.resp, item.txn.data[0]))
        endtask

        virtual task body();
            transfer(0, 'h00, 0, RESP_OKAY, 'h41500200);
            transfer(1, 'h10, 'h11223344, RESP_OKAY, 0);
            transfer(0, 'h10, 0, RESP_OKAY, 'h11223344);
            transfer(1, 'h1010, 'hffffffff, RESP_SLVERR, 0);
            transfer(0, 'h11, 0, RESP_SLVERR, 0);
            transfer(0, 'h14, 0, RESP_SLVERR, 0);
            transfer(0, 'h10, 0, RESP_OKAY, 'h11223344);
        endtask
    endclass

    class portfolio_monitor extends uvm_subscriber #(amba_seq_item);
        `uvm_component_utils(portfolio_monitor)
        int observed;
        function new(string name, uvm_component parent); super.new(name, parent); endfunction
        function void write(amba_seq_item item);
            observed++;
        endfunction
    endclass

    class portfolio_test extends uvm_test;
        `uvm_component_utils(portfolio_test)
        apb_driver driver;
        apb_sequencer sequencer;
        apb_monitor monitor;
        portfolio_monitor observer;
        virtual amba_vip_if vif;

        function new(string name, uvm_component parent); super.new(name, parent); endfunction

        function void build_phase(uvm_phase phase);
            super.build_phase(phase);
            if (!uvm_config_db#(virtual amba_vip_if)::get(this, "", "vif", vif))
                `uvm_fatal("NOVIF", "missing APB interface")
            driver = apb_driver::type_id::create("driver", this);
            sequencer = apb_sequencer::type_id::create("sequencer", this);
            monitor = apb_monitor::type_id::create("monitor", this);
            observer = portfolio_monitor::type_id::create("observer", this);
        endfunction

        function void connect_phase(uvm_phase phase);
            super.connect_phase(phase);
            driver.seq_item_port.connect(sequencer.seq_item_export);
            monitor.beat_ap.connect(observer.analysis_export);
        endfunction

        task run_phase(uvm_phase phase);
            portfolio_sequence seq = portfolio_sequence::type_id::create("seq");
            phase.raise_objection(this);
            wait (vif.aresetn === 1'b1);
            seq.start(sequencer);
            repeat (2) @(posedge vif.aclk);
            if (observer.observed != 7)
                `uvm_fatal("APB_COUNT", $sformatf("expected 7 transfers, got %0d", observer.observed))
            `uvm_info("PORTFOLIO", "APB_PORTFOLIO_UVM_PASS checks=7", UVM_NONE)
            phase.drop_objection(this);
        endtask
    endclass
endpackage

module apb_portfolio_uvm_tb;
    timeunit 1ns;
    timeprecision 1ps;
    import uvm_pkg::*;
    import apb_portfolio_test_pkg::*;
    logic clk = 0, rst_n = 0;
    wire [31:0] read_data;
    amba_vip_if vif(clk, rst_n);

    always #5ns clk = ~clk;
    apb_portfolio_bridge bridge (
        .clk(clk), .rst_n(rst_n), .select(vif.psel), .enable(vif.penable),
        .write(vif.pwrite), .address(vif.paddr), .write_data(vif.pwdata),
        .strobe(vif.pstrb), .read_data(read_data), .ready(vif.pready), .error(vif.pslverr)
    );
    assign vif.prdata = read_data ^ (($test$plusargs("INJECT_BAD_ID") &&
                                      vif.paddr == 0) ? 32'd1 : 32'd0);
    apb_compliance_checker_sva checker_inst(vif);

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
    end
    initial begin
        uvm_config_db#(virtual amba_vip_if)::set(null, "*", "vif", vif);
        run_test("portfolio_test");
    end
    initial begin
        #2000ns;
        $fatal(1, "UVM timeout");
    end
endmodule
