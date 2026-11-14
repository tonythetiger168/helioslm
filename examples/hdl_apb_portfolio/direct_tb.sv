// SPDX-License-Identifier: Apache-2.0
`timescale 1ns/1ps
module apb_portfolio_direct_tb;
    logic clk = 0, rst_n = 0, select = 0, enable = 0, write = 0;
    logic [31:0] address = 0, write_data = 0;
    logic [3:0] strobe = 0;
    wire [31:0] read_data;
    wire ready, error;
    int checks = 0;

    always #5 clk = ~clk;
    apb_portfolio_bridge bridge (.*);

    task automatic transfer(input bit wr, input logic [31:0] addr, data,
                            input logic [3:0] mask, input bit expected_error,
                            input logic [31:0] expected_data);
        @(negedge clk);
        select = 1; enable = 0; write = wr;
        address = addr; write_data = data; strobe = mask;
        #1;
        if (ready !== 0 || error !== 0) $fatal(1, "response during SETUP");
        @(negedge clk);
        enable = 1;
        #1;
        if (ready !== 1 || error !== expected_error)
            $fatal(1, "bad response at %h: ready=%b error=%b", addr, ready, error);
        if (!wr && read_data !== expected_data)
            $fatal(1, "bad read at %h: %h expected %h", addr, read_data, expected_data);
        checks++;
        @(negedge clk);
        select = 0; enable = 0;
    endtask

    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1;
        transfer(0, 'h00, 0, 0, 0, 32'h41500200);
        transfer(0, 'h10, 0, 0, 0, 0);
        transfer(1, 'h10, 'h11223344, 'hf, 0, 0);
        transfer(1, 'h10, 'haabbccdd, 'h5, 0, 0);
        transfer(0, 'h10, 0, 0, 0, 'h11bb33dd);
        transfer(1, 'h1010, 'hffffffff, 'hf, 1, 0);
        transfer(0, 'h1010, 0, 0, 1, 0);
        transfer(0, 'h11, 0, 0, 1, 0);
        transfer(0, 'h14, 0, 0, 1, 0);
        transfer(0, 'h10, 0, 0, 0, 'h11bb33dd);
        $display("APB_PORTFOLIO_DIRECT_PASS checks=%0d", checks);
        $finish;
    end

    initial begin
        #2000;
        $fatal(1, "APB timeout");
    end
endmodule
